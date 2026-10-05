#include "jevdb.hpp"
#include "duckdb/common/enums/cte_materialize.hpp"
#include "duckdb/optimizer/column_binding_replacer.hpp"
#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_conjunction_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_cross_product.hpp"
#include "duckdb/planner/operator/logical_cteref.hpp"
#include "duckdb/planner/operator/logical_distinct.hpp"
#include "duckdb/planner/operator/logical_materialized_cte.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include <set>

namespace duckdb {
namespace jevdb {

namespace {

using Sources = std::set<vector<idx_t>>;
using BindingKey = std::pair<idx_t, idx_t>;
using Lineage = std::map<BindingKey, Sources>;

static BindingKey Key(const ColumnBinding &binding) {
	return {binding.table_index, binding.column_index};
}

static Sources InputSources(const Expression &expression, const Lineage &lineage) {
	Sources result;
	ExpressionIterator::VisitExpression<BoundColumnRefExpression>(
	    expression, [&](const BoundColumnRefExpression &column) {
		    auto entry = lineage.find(Key(column.binding));
		    if (entry == lineage.end()) {
			    result.insert(vector<idx_t> {column.binding.table_index});
		    } else {
			    result.insert(entry->second.begin(), entry->second.end());
		    }
	    });
	return result;
}

static void ReadConjuncts(const Expression &expression, vector<const Expression *> &result) {
	if (expression.type == ExpressionType::CONJUNCTION_AND) {
		for (auto &child : expression.Cast<BoundConjunctionExpression>().children) {
			ReadConjuncts(*child, result);
		}
	} else {
		result.push_back(&expression);
	}
}

static bool IsKeys(const Expression &expression) {
	return expression.GetExpressionClass() == ExpressionClass::BOUND_FUNCTION &&
	       expression.Cast<BoundFunctionExpression>().function.name == "jev_keys_overlap";
}

// An explicit key screen belongs to the same inputs when its two sides come
// from the same relation aliases as the semantic inputs. Projection and CTE
// aliases are traced below, rather than matching arbitrary keys in an ancestor.
class SBFOptimizer {
public:
	explicit SBFOptimizer(Optimizer &optimizer) : optimizer(optimizer) {
	}
	void Optimize(unique_ptr<LogicalOperator> &plan) {
		BuildLineage(*plan);
		Transform(plan);
		ApplyFixes(*plan);
	}
	void Reduce(unique_ptr<LogicalOperator> &plan) {
		// A session that turns off native join ordering or filter pushdown keeps the written order here as well.
		if (optimizer.OptimizerDisabled(OptimizerType::FILTER_PUSHDOWN) ||
		    optimizer.OptimizerDisabled(OptimizerType::JOIN_ORDER))
			return;
		ReduceInputs(plan);
		ApplyFixes(*plan);
	}
	void Prepare(unique_ptr<LogicalOperator> &plan) {
		if (optimizer.OptimizerDisabled(OptimizerType::FILTER_PUSHDOWN) ||
		    optimizer.OptimizerDisabled(OptimizerType::JOIN_ORDER))
			return;
		BuildLineage(*plan);
		PrepareRegions(plan);
	}

private:
	struct Fix {
		LogicalOperator *boundary;
		vector<ColumnBinding> before;
		vector<ColumnBinding> after;
	};
	Optimizer &optimizer;
	Lineage lineage;
	std::map<idx_t, vector<Sources>> cte_sources;
	vector<Fix> fixes;

	struct Region {
		Region(bool safe = true, bool keyed_any = false, bool comparison_above_any = false)
		    : safe(safe), keyed_any(keyed_any), comparison_above_any(comparison_above_any) {
		}
		bool safe;
		bool keyed_any;
		bool comparison_above_any;
	};

	Region ReadRegion(LogicalOperator &op) {
		Region region;
		vector<const Expression *> conjuncts;
		if (op.type == LogicalOperatorType::LOGICAL_FILTER) {
			for (auto &expression : op.expressions)
				ReadConjuncts(*expression, conjuncts);
		} else if (op.type == LogicalOperatorType::LOGICAL_ANY_JOIN) {
			auto &join = op.Cast<LogicalAnyJoin>();
			if (join.join_type != JoinType::INNER)
				return {false, false, false};
			ReadConjuncts(*join.condition, conjuncts);
		} else if (op.type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
			auto &join = op.Cast<LogicalComparisonJoin>();
			if (join.join_type != JoinType::INNER)
				return {false, false, false};
			for (auto &condition : join.conditions) {
				if (condition.left->IsVolatile() || condition.right->IsVolatile() || ContainsHolds(*condition.left) ||
				    ContainsHolds(*condition.right))
					return {false, false, false};
			}
		} else if (op.type != LogicalOperatorType::LOGICAL_CROSS_PRODUCT) {
			// Keep the change within the observed scan/INNER component;
			// projections, CTEs, outer joins and other boundaries exclude it.
			region.safe = (op.type == LogicalOperatorType::LOGICAL_GET ||
			               op.type == LogicalOperatorType::LOGICAL_EXPRESSION_GET) &&
			              op.children.empty();
			return region;
		}
		if (op.HasProjectionMap())
			return {false, false, false};
		vector<const BoundFunctionExpression *> keys;
		CollectKeys(op, keys);
		for (auto expression : conjuncts) {
			if (expression->IsVolatile())
				return {false, false, false};
			if (IsHolds(*expression)) {
				if (!RelatedKeys(expression->Cast<BoundFunctionExpression>(), keys))
					return {false, false, false};
				if (op.type == LogicalOperatorType::LOGICAL_ANY_JOIN)
					region.keyed_any = true;
			} else if (ContainsHolds(*expression)) {
				return {false, false, false};
			}
		}
		if (op.type == LogicalOperatorType::LOGICAL_ANY_JOIN && !region.keyed_any)
			return {false, false, false};
		for (auto &child : op.children) {
			auto below = ReadRegion(*child);
			region.safe &= below.safe;
			region.keyed_any |= below.keyed_any;
			region.comparison_above_any |= below.comparison_above_any;
			if (op.type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN && below.keyed_any)
				region.comparison_above_any = true;
		}
		return region;
	}

	bool HasSemanticPredicate(LogicalOperator &op) {
		for (auto &expression : op.expressions) {
			if (ContainsHolds(*expression))
				return true;
		}
		if (op.type == LogicalOperatorType::LOGICAL_ANY_JOIN && ContainsHolds(*op.Cast<LogicalAnyJoin>().condition))
			return true;
		for (auto &child : op.children) {
			if (HasSemanticPredicate(*child))
				return true;
		}
		return false;
	}

	void MoveOrdinaryJoins(unique_ptr<LogicalOperator> &op) {
		for (auto &child : op->children)
			MoveOrdinaryJoins(child);
		auto cross = op->type == LogicalOperatorType::LOGICAL_CROSS_PRODUCT;
		if (!cross && op->type != LogicalOperatorType::LOGICAL_COMPARISON_JOIN)
			return;
		unordered_set<idx_t> references;
		if (!cross) {
			for (auto &condition : op->Cast<LogicalComparisonJoin>().conditions) {
				LogicalJoin::GetExpressionBindings(*condition.left, references);
				LogicalJoin::GetExpressionBindings(*condition.right, references);
			}
		}
		for (idx_t child = 0; child < 2; child++) {
			if (op->children[child]->type != LogicalOperatorType::LOGICAL_ANY_JOIN)
				continue;
			if (cross && HasSemanticPredicate(*op->children[1 - child]))
				continue;
			auto &semantic = *op->children[child];
			unordered_set<idx_t> left, right;
			LogicalJoin::GetTableReferences(*semantic.children[0], left);
			LogicalJoin::GetTableReferences(*semantic.children[1], right);
			bool uses_left = false, uses_right = false;
			for (auto binding : references) {
				uses_left |= left.count(binding) != 0;
				uses_right |= right.count(binding) != 0;
			}
			if (!cross && uses_left == uses_right)
				continue;
			// Move the ordinary join into the one semantic input referenced
			// by all its conditions, retaining the semantic join's boundary.
			// ReadRegion proved both joins are INNER and have no projection
			// maps. ColumnBinding identities and bag multiplicities remain.
			// An ordinary CROSS input belongs once on the first semantic
			// input side. Native pushdown can then screen that side using its
			// parameters; conditions involving other sides stay above it.
			auto side = cross || uses_left ? idx_t(0) : idx_t(1);
			auto boundary = std::move(op->children[child]);
			op->children[child] = std::move(boundary->children[side]);
			boundary->children[side] = std::move(op);
			MoveOrdinaryJoins(boundary->children[side]);
			op = std::move(boundary);
			return;
		}
	}

	static bool SemanticJoin(LogicalOperator &op) {
		if (op.type != LogicalOperatorType::LOGICAL_ANY_JOIN)
			return false;
		auto &join = op.Cast<LogicalAnyJoin>();
		return (join.join_type == JoinType::INNER || join.join_type == JoinType::SEMI ||
		        join.join_type == JoinType::ANTI) &&
		       ContainsHolds(*join.condition);
	}

	// A correlated EXISTS that planning could not turn into a plain join stays a duplicate-eliminating join; it
	// restricts its left input like a SEMI or ANTI join and moves with its right side.
	static bool OrdinaryJoin(LogicalOperator &op) {
		if (op.type != LogicalOperatorType::LOGICAL_COMPARISON_JOIN &&
		    op.type != LogicalOperatorType::LOGICAL_DELIM_JOIN)
			return false;
		auto &join = op.Cast<LogicalComparisonJoin>();
		if (op.type == LogicalOperatorType::LOGICAL_DELIM_JOIN)
			return !join.delim_flipped && (join.join_type == JoinType::SEMI || join.join_type == JoinType::ANTI);
		return join.join_type == JoinType::INNER || join.join_type == JoinType::SEMI ||
		       join.join_type == JoinType::ANTI;
	}

	// The child of an ordinary join through which a semantic join is reached by ordinary joins only.
	idx_t TowerChild(LogicalOperator &op) {
		auto found = DConstants::INVALID_INDEX;
		for (idx_t child = 0; child < 2; child++) {
			auto &below = *op.children[child];
			if (SemanticJoin(below) || (OrdinaryJoin(below) && TowerChild(below) != DConstants::INVALID_INDEX)) {
				if (found != DConstants::INVALID_INDEX)
					return DConstants::INVALID_INDEX;
				found = child;
			}
		}
		return found;
	}

	// An ordinary join above a semantic join can be evaluated inside one semantic input when every condition reads
	// that input alone on the semantic side, and its other input holds no semantic predicate. INNER, SEMI and ANTI
	// joins keep their result when moved: the rows of that input without a partner would not reach the output
	// either way. Returns the semantic input, or INVALID_INDEX.
	idx_t ReducedInput(LogicalOperator &ordinary, idx_t child, LogicalOperator &semantic) {
		auto &join = ordinary.Cast<LogicalComparisonJoin>();
		if ((join.join_type != JoinType::INNER && child != 0) || HasSemanticPredicate(*ordinary.children[1 - child]))
			return DConstants::INVALID_INDEX;
		unordered_set<idx_t> references;
		for (auto &condition : join.conditions) {
			if (condition.left->IsVolatile() || condition.right->IsVolatile())
				return DConstants::INVALID_INDEX;
			LogicalJoin::GetExpressionBindings(child == 0 ? *condition.left : *condition.right, references);
		}
		for (auto &column : join.duplicate_eliminated_columns)
			LogicalJoin::GetExpressionBindings(*column, references);
		unordered_set<idx_t> left, right;
		LogicalJoin::GetTableReferences(*semantic.children[0], left);
		LogicalJoin::GetTableReferences(*semantic.children[1], right);
		bool uses_left = false, uses_right = false;
		for (auto table : references) {
			if (!left.count(table) && !right.count(table))
				return DConstants::INVALID_INDEX;
			uses_left |= left.count(table) != 0;
			uses_right |= right.count(table) != 0;
		}
		if (uses_left == uses_right)
			return DConstants::INVALID_INDEX;
		// Only the left input of a semantic SEMI or ANTI join is part of its output.
		if (uses_right && semantic.Cast<LogicalAnyJoin>().join_type != JoinType::INNER)
			return DConstants::INVALID_INDEX;
		return uses_left ? 0 : 1;
	}

	// Relational reduction of semantic inputs. Planning leaves a semantic join below the ordinary joins that follow
	// it, so every row of its inputs is paired and judged before those joins discard rows. Here the ordinary joins
	// directly above a semantic join move into the input they restrict, innermost first. The semantic join then
	// receives only rows that still have partners in the relations behind that input, which is what the inward pass
	// of Yannakakis' algorithm leaves at that relation. The output columns of the rewritten subtree are restored by
	// a projection, as the operators above address them by position through their projection maps.
	void ReduceInputs(unique_ptr<LogicalOperator> &op) {
		if (OrdinaryJoin(*op) && TowerChild(*op) != DConstants::INVALID_INDEX) {
			vector<LogicalOperator *> tower;
			vector<idx_t> path;
			for (auto current = op.get(); !SemanticJoin(*current); current = current->children[path.back()].get()) {
				tower.push_back(current);
				path.push_back(TowerChild(*current));
			}
			auto &lowest = *tower.back();
			if (ReducedInput(lowest, path.back(), *lowest.children[path.back()]) != DConstants::INVALID_INDEX) {
				op->ResolveOperatorTypes();
				auto bindings = op->GetColumnBindings();
				auto types = op->types;
				// Take the tower apart, then rebuild it around the semantic join from the inside.
				vector<unique_ptr<LogicalOperator>> joins;
				auto top = std::move(op);
				for (auto child : path) {
					auto next = std::move(top->children[child]);
					joins.push_back(std::move(top));
					top = std::move(next);
				}
				auto &semantic = top->Cast<LogicalAnyJoin>();
				semantic.left_projection_map.clear();
				semantic.right_projection_map.clear();
				for (idx_t level = joins.size(); level-- > 0;) {
					auto &join = joins[level]->Cast<LogicalComparisonJoin>();
					join.left_projection_map.clear();
					join.right_projection_map.clear();
					auto side =
					    top.get() == &semantic ? ReducedInput(join, path[level], semantic) : DConstants::INVALID_INDEX;
					if (side == DConstants::INVALID_INDEX) {
						join.children[path[level]] = std::move(top);
						top = std::move(joins[level]);
					} else {
						join.children[path[level]] = std::move(semantic.children[side]);
						semantic.children[side] = std::move(joins[level]);
					}
				}
				auto index = optimizer.binder.GenerateTableIndex();
				vector<unique_ptr<Expression>> output;
				Fix fix {nullptr, bindings, {}};
				for (idx_t column = 0; column < bindings.size(); column++) {
					output.push_back(make_uniq<BoundColumnRefExpression>(types[column], bindings[column]));
					fix.after.emplace_back(index, column);
				}
				auto result = make_uniq<LogicalProjection>(index, std::move(output));
				result->children.push_back(std::move(top));
				fix.boundary = result.get();
				fixes.push_back(std::move(fix));
				op = std::move(result);
			}
		}
		for (auto &child : op->children)
			ReduceInputs(child);
	}

	void PrepareRegions(unique_ptr<LogicalOperator> &op) {
		auto region = ReadRegion(*op);
		if (region.safe && region.comparison_above_any) {
			// Let the normal built-in rules plan each ordinary input subtree,
			// retaining the semantic edges instead of merging all predicates
			// above one large combination of independent business inputs.
			MoveOrdinaryJoins(op);
			return;
		}
		for (auto &child : op->children)
			PrepareRegions(child);
	}

	void BuildLineage(LogicalOperator &op) {
		if (op.type == LogicalOperatorType::LOGICAL_MATERIALIZED_CTE) {
			BuildLineage(*op.children[0]);
			auto &cte = op.Cast<LogicalMaterializedCTE>();
			for (auto &binding : op.children[0]->GetColumnBindings()) {
				cte_sources[cte.table_index].push_back(lineage[Key(binding)]);
			}
			BuildLineage(*op.children[1]);
		} else {
			for (auto &child : op.children)
				BuildLineage(*child);
		}
		if (op.type == LogicalOperatorType::LOGICAL_PROJECTION) {
			auto &projection = op.Cast<LogicalProjection>();
			for (idx_t column = 0; column < projection.expressions.size(); column++) {
				lineage[{projection.table_index, column}] = InputSources(*projection.expressions[column], lineage);
			}
		} else if (op.type == LogicalOperatorType::LOGICAL_CTE_REF) {
			auto &ref = op.Cast<LogicalCTERef>();
			auto producer = cte_sources.find(ref.cte_index);
			for (idx_t column = 0; column < ref.chunk_types.size(); column++) {
				Sources sources;
				if (producer == cte_sources.end()) {
					sources.insert(vector<idx_t> {ref.table_index});
				} else {
					// Each consumer is a separate relation alias, even when its
					// producer is shared. Retain the column's producer lineage
					// to distinguish sides of a joined/materialized input.
					for (auto source : producer->second[column]) {
						source.insert(source.begin(), ref.table_index);
						sources.insert(std::move(source));
					}
				}
				lineage[{ref.table_index, column}] = std::move(sources);
			}
		} else {
			for (auto &binding : op.GetColumnBindings()) {
				lineage.emplace(Key(binding), Sources {vector<idx_t> {binding.table_index}});
			}
		}
	}

	void CollectKeys(LogicalOperator &op, vector<const BoundFunctionExpression *> &keys) {
		vector<const Expression *> conjuncts;
		if (op.type == LogicalOperatorType::LOGICAL_FILTER) {
			for (auto &expression : op.expressions)
				ReadConjuncts(*expression, conjuncts);
		} else if (op.type == LogicalOperatorType::LOGICAL_ANY_JOIN) {
			auto &join = op.Cast<LogicalAnyJoin>();
			if (join.join_type != JoinType::INNER && join.join_type != JoinType::SEMI &&
			    join.join_type != JoinType::ANTI)
				return;
			ReadConjuncts(*join.condition, conjuncts);
		} else if (op.type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
			auto type = op.Cast<LogicalComparisonJoin>().join_type;
			if (type != JoinType::INNER && type != JoinType::SEMI)
				return;
		} else if (op.type != LogicalOperatorType::LOGICAL_PROJECTION &&
		           op.type != LogicalOperatorType::LOGICAL_CROSS_PRODUCT) {
			return;
		}
		for (auto expression : conjuncts) {
			if (IsKeys(*expression))
				keys.push_back(&expression->Cast<BoundFunctionExpression>());
		}
		for (auto &child : op.children)
			CollectKeys(*child, keys);
	}

	bool RelatedKeys(const BoundFunctionExpression &holds, const vector<const BoundFunctionExpression *> &keys) {
		auto left = InputSources(*holds.children[1], lineage);
		auto &data = holds.bind_info->Cast<HoldsData>();
		auto right = data.pairwise ? InputSources(*holds.children[2], lineage) : Sources {};
		for (auto key : keys) {
			auto key_left = InputSources(*key->children[0], lineage);
			auto key_right = InputSources(*key->children[1], lineage);
			if (key_left.empty() || key_right.empty())
				continue;
			if (data.pairwise) {
				if ((left == key_left && right == key_right) || (left == key_right && right == key_left))
					return true;
			} else {
				key_left.insert(key_right.begin(), key_right.end());
				if (left == key_left)
					return true;
			}
		}
		return false;
	}

	void ApplyFixes(LogicalOperator &op) {
		for (auto &fix : fixes) {
			ColumnBindingReplacer replacer;
			replacer.stop_operator = fix.boundary;
			for (idx_t column = 0; column < fix.before.size(); column++) {
				replacer.replacement_bindings.emplace_back(fix.before[column], fix.after[column]);
			}
			replacer.VisitOperator(op);
		}
	}

	ColumnBinding CurrentBinding(ColumnBinding binding) {
		for (auto &fix : fixes) {
			for (idx_t column = 0; column < fix.before.size(); column++) {
				if (binding == fix.before[column])
					binding = fix.after[column];
			}
		}
		return binding;
	}

	unique_ptr<LogicalOperator> Factorize(unique_ptr<LogicalOperator> producer, unique_ptr<Expression> predicate) {
		producer->ResolveOperatorTypes();
		auto bindings = producer->GetColumnBindings();
		auto types = producer->types;
		auto &holds = predicate->Cast<BoundFunctionExpression>();
		auto input_count = holds.bind_info->Cast<HoldsData>().pairwise ? idx_t(2) : idx_t(1);
		vector<LogicalType> input_types;
		vector<idx_t> input_columns;
		auto stored_types = types;
		vector<unique_ptr<Expression>> stored;
		for (idx_t column = 0; column < bindings.size(); column++) {
			stored.push_back(make_uniq<BoundColumnRefExpression>(types[column], bindings[column]));
		}
		for (idx_t input = 0; input < input_count; input++) {
			auto &expression = *holds.children[input + 1];
			input_types.push_back(expression.return_type);
			auto position = DConstants::INVALID_INDEX;
			if (expression.GetExpressionClass() == ExpressionClass::BOUND_COLUMN_REF) {
				auto binding = expression.Cast<BoundColumnRefExpression>().binding;
				for (idx_t column = 0; column < bindings.size(); column++) {
					if (bindings[column] == binding)
						position = column;
				}
			}
			if (position == DConstants::INVALID_INDEX) {
				position = stored.size();
				stored.push_back(expression.Copy());
				stored_types.push_back(expression.return_type);
			}
			input_columns.push_back(position);
		}
		auto stored_index = optimizer.binder.GenerateTableIndex();
		auto stored_producer = make_uniq<LogicalProjection>(stored_index, std::move(stored));
		stored_producer->children.push_back(std::move(producer));
		vector<string> names;
		for (idx_t column = 0; column < stored_types.size(); column++)
			names.push_back("jev_sbf_col_" + std::to_string(column));
		auto cte_index = optimizer.binder.GenerateTableIndex();
		auto full_index = optimizer.binder.GenerateTableIndex();
		auto input_index = optimizer.binder.GenerateTableIndex();
		auto distinct_index = optimizer.binder.GenerateTableIndex();
		auto full = make_uniq<LogicalCTERef>(full_index, cte_index, stored_types, names,
		                                     CTEMaterialize::CTE_MATERIALIZE_ALWAYS);
		auto inputs = make_uniq<LogicalCTERef>(input_index, cte_index, stored_types, names,
		                                       CTEMaterialize::CTE_MATERIALIZE_ALWAYS);
		vector<unique_ptr<Expression>> tuple;
		vector<unique_ptr<Expression>> targets;
		for (idx_t input = 0; input < input_count; input++) {
			tuple.push_back(make_uniq<BoundColumnRefExpression>(input_types[input],
			                                                    ColumnBinding(input_index, input_columns[input])));
			targets.push_back(
			    make_uniq<BoundColumnRefExpression>(input_types[input], ColumnBinding(distinct_index, input)));
			holds.children[input + 1] =
			    make_uniq<BoundColumnRefExpression>(input_types[input], ColumnBinding(distinct_index, input));
		}
		auto projection = make_uniq<LogicalProjection>(distinct_index, std::move(tuple));
		projection->children.push_back(std::move(inputs));
		auto distinct = make_uniq<LogicalDistinct>(std::move(targets), DistinctType::DISTINCT);
		distinct->children.push_back(std::move(projection));
		auto accepted = make_uniq<LogicalFilter>(std::move(predicate));
		accepted->children.push_back(std::move(distinct));
		auto replay = make_uniq<LogicalComparisonJoin>(JoinType::SEMI);
		for (idx_t input = 0; input < input_count; input++) {
			JoinCondition condition;
			condition.comparison = ExpressionType::COMPARE_NOT_DISTINCT_FROM;
			condition.left = make_uniq<BoundColumnRefExpression>(input_types[input],
			                                                     ColumnBinding(full_index, input_columns[input]));
			condition.right =
			    make_uniq<BoundColumnRefExpression>(input_types[input], ColumnBinding(distinct_index, input));
			replay->conditions.push_back(std::move(condition));
		}
		for (idx_t column = 0; column < types.size(); column++)
			replay->left_projection_map.push_back(column);
		replay->children.push_back(std::move(full));
		replay->children.push_back(std::move(accepted));
		auto result = make_uniq<LogicalMaterializedCTE>("jev_sbf_" + std::to_string(cte_index), cte_index,
		                                                stored_types.size(), std::move(stored_producer),
		                                                std::move(replay), CTEMaterialize::CTE_MATERIALIZE_ALWAYS);
		Fix fix {result.get(), std::move(bindings), {}};
		for (idx_t column = 0; column < types.size(); column++)
			fix.after.emplace_back(full_index, column);
		fixes.push_back(std::move(fix));
		return result;
	}

	void Transform(unique_ptr<LogicalOperator> &op) {
		vector<const Expression *> conjuncts;
		bool supported = op->type == LogicalOperatorType::LOGICAL_FILTER;
		bool positive_region = supported;
		if (supported) {
			for (auto &expression : op->expressions)
				ReadConjuncts(*expression, conjuncts);
		} else if (op->type == LogicalOperatorType::LOGICAL_ANY_JOIN) {
			auto &join = op->Cast<LogicalAnyJoin>();
			supported = join.join_type == JoinType::INNER;
			positive_region = supported || join.join_type == JoinType::SEMI || join.join_type == JoinType::ANTI;
			if (positive_region)
				ReadConjuncts(*join.condition, conjuncts);
		}
		vector<const BoundFunctionExpression *> keys;
		if (positive_region)
			CollectKeys(*op, keys);
		std::set<const Expression *> factorized;
		for (auto expression : conjuncts) {
			if (IsHolds(*expression)) {
				if (RelatedKeys(expression->Cast<BoundFunctionExpression>(), keys))
					factorized.insert(expression);
			} else if (ContainsHolds(*expression)) {
				supported = false;
				positive_region = false;
			}
		}
		if (positive_region) {
			for (auto expression : factorized) {
				expression->Cast<BoundFunctionExpression>().bind_info->Cast<HoldsData>().stable_inputs = true;
			}
		}
		for (auto &child : op->children) {
			Transform(child);
			ApplyFixes(*op);
		}
		if (op->type == LogicalOperatorType::LOGICAL_ANY_JOIN && CanUseKeyCandidates(op->Cast<LogicalAnyJoin>())) {
			// Keep this boundary for the join's direct pair generation.
			// Key columns may belong to an attached-key input rather than the
			// semantic text's base table. Its first answer, including failure,
			// must still be reused by later consumers in this query.
			for (auto expression : conjuncts) {
				if (IsHolds(*expression))
					expression->Cast<BoundFunctionExpression>().bind_info->Cast<HoldsData>().stable_inputs = true;
			}
			return;
		}
		if (!supported || factorized.empty())
			return;
		op->ResolveOperatorTypes();
		auto original_bindings = op->GetColumnBindings();
		auto original_types = op->types;
		vector<unique_ptr<Expression>> expressions;
		unique_ptr<LogicalOperator> producer;
		if (op->type == LogicalOperatorType::LOGICAL_FILTER) {
			expressions = std::move(op->expressions);
			producer = std::move(op->children[0]);
		} else {
			auto &join = op->Cast<LogicalAnyJoin>();
			expressions.push_back(std::move(join.condition));
			producer = LogicalCrossProduct::Create(std::move(op->children[0]), std::move(op->children[1]));
		}
		LogicalFilter::SplitPredicates(expressions);
		vector<unique_ptr<Expression>> semantic;
		auto screen = make_uniq<LogicalFilter>();
		for (auto &expression : expressions) {
			if (IsHolds(*expression))
				semantic.push_back(std::move(expression));
			else
				screen->expressions.push_back(std::move(expression));
		}
		if (!screen->expressions.empty()) {
			screen->children.push_back(std::move(producer));
			producer = std::move(screen);
		}
		for (auto &expression : semantic) {
			auto stage = make_uniq<LogicalFilter>(std::move(expression));
			stage->children.push_back(std::move(producer));
			ApplyFixes(*stage);
			if (factorized.count(stage->expressions[0].get())) {
				producer = Factorize(std::move(stage->children[0]), std::move(stage->expressions[0]));
			} else {
				producer = std::move(stage);
			}
		}
		// Restore precisely the original output columns. Join/filter projection
		// maps can omit the semantic input columns after their last use.
		auto output_index = optimizer.binder.GenerateTableIndex();
		vector<unique_ptr<Expression>> output;
		vector<ColumnBinding> output_bindings;
		for (idx_t column = 0; column < original_bindings.size(); column++) {
			auto binding = CurrentBinding(original_bindings[column]);
			output_bindings.push_back(binding);
			output.push_back(make_uniq<BoundColumnRefExpression>(original_types[column], binding));
		}
		auto result = make_uniq<LogicalProjection>(output_index, std::move(output));
		result->children.push_back(std::move(producer));
		Fix fix {result.get(), std::move(output_bindings), {}};
		for (idx_t column = 0; column < original_types.size(); column++)
			fix.after.emplace_back(output_index, column);
		fixes.push_back(std::move(fix));
		op = std::move(result);
	}
};

} // namespace

void OptimizeSBF(Optimizer &optimizer, unique_ptr<LogicalOperator> &plan) {
	SBFOptimizer(optimizer).Optimize(plan);
}

void PrepareSBF(Optimizer &optimizer, unique_ptr<LogicalOperator> &plan) {
	SBFOptimizer(optimizer).Prepare(plan);
}

void ReduceSemanticInputs(Optimizer &optimizer, unique_ptr<LogicalOperator> &plan) {
	SBFOptimizer(optimizer).Reduce(plan);
}

} // namespace jevdb
} // namespace duckdb
