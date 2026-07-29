#ifndef TERMINAL_BRANCH_LENGTH_HPP
#define TERMINAL_BRANCH_LENGTH_HPP

#include "jc69_msc_probabilities.hpp"
#include "threadpool.hpp"

namespace terminal_branch_length {

using std::array;
using std::size_t;
using std::vector;

struct FitResult {
	bool success = false;
	long double logLikelihood =
		-std::numeric_limits<long double>::infinity();
	long double theta = 0;
	array<long double, 2> speciesAges{};
};

template<typename PatternCounts> class Estimator {
	static constexpr array<array<unsigned char, 3>, 5> CLASSES = {{
		{{0, 0, 0}}, {{0, 0, 1}}, {{0, 1, 0}},
		{{0, 1, 1}}, {{0, 1, 2}}
	}};

	static long double correctedDistance(
		long double mismatchFraction
	) noexcept {
		if (mismatchFraction < 0 || mismatchFraction >= 0.75L)
			return -1;
		long double remaining = 1 - 4 * mismatchFraction / 3;
		return remaining > 0 ? -0.75L * std::log(remaining) : -1;
	}

	static array<long double, 3> pairwiseDistances(
		PatternCounts const& counts
	) noexcept {
		long double total = 0;
		for (long double count : counts) total += count;

		array<long double, 3> distances{};
		size_t iPair = 0;
		for (size_t i = 0; i < 3; ++i) {
			for (size_t j = i + 1; j < 3; ++j) {
				long double mismatch = 0;
				for (size_t k = 0; k < CLASSES.size(); ++k)
					if (CLASSES[k][i] != CLASSES[k][j])
						mismatch += counts[k];
				distances[iPair++] =
					correctedDistance(mismatch / total);
			}
		}
		return distances;
	}

	static vector<long double> initialParameters(
		PatternCounts const& counts,
		std::optional<long double> fixedTheta
	) noexcept {
		// Pair order is AB, AC, BC. As in the quartet estimator, use
		// representative leaves A/B for s1 and A/C for s2.
		array<long double, 3> distances = pairwiseDistances(counts);
		long double validSum = 0;
		size_t nValid = 0;
		for (long double value : distances) {
			if (value >= 0) {
				validSum += value;
				++nValid;
			}
		}
		long double fallback =
			nValid ? std::max(validSum / nValid, 0.05L) : 0.1L;
		for (long double& value : distances)
			if (value < 0) value = fallback;

		long double age1 = std::max(distances[0] / 2, 1e-6L);
		long double age2 =
			std::max(distances[1] / 2, age1 + 1e-6L);
		vector<long double> result = {
			std::log(age1), std::log(age2 - age1)
		};
		if (!fixedTheta)
			result.push_back(std::log(std::max(fallback, 1e-3L)));
		return result;
	}

	static bool unpack(
		vector<long double> const& params,
		std::optional<long double> fixedTheta,
		array<long double, 2>& speciesAges,
		long double& theta
	) noexcept {
		const size_t expected = fixedTheta ? 2 : 3;
		if (params.size() != expected) return false;
		constexpr long double LOG_MIN = -30;
		constexpr long double LOG_MAX = 10;
		array<long double, 3> positive{};
		for (size_t i = 0; i < params.size(); ++i) {
			if (params[i] < LOG_MIN || params[i] > LOG_MAX)
				return false;
			positive[i] = std::exp(params[i]);
		}
		speciesAges[0] = positive[0];
		speciesAges[1] = speciesAges[0] + positive[1];
		theta = fixedTheta ? *fixedTheta : positive[2];
		return theta > 0 && std::isfinite(theta);
	}

	static long double objective(
		PatternCounts const& counts,
		vector<long double> const& params,
		std::optional<long double> fixedTheta
	) noexcept {
		array<long double, 2> ages{};
		long double theta = 0;
		if (!unpack(params, fixedTheta, ages, theta))
			return std::numeric_limits<long double>::infinity();

		auto probabilities =
			branch_length::jc69_msc::triplet(
				ages[0], ages[1], theta
			);
		long double logLikelihood = 0;
		for (size_t i = 0; i < counts.size(); ++i) {
			long double probability = probabilities[i];
			if (!(probability > 0))
				return std::numeric_limits<long double>::infinity();
			if (counts[i] != 0)
				logLikelihood += counts[i] * std::log(probability);
		}
		return -logLikelihood;
	}

	static std::pair<vector<long double>, long double> nelderMead(
		PatternCounts const& counts,
		vector<long double> const& initial,
		std::optional<long double> fixedTheta,
		size_t maxIterations = 200
	) noexcept {
		constexpr long double STEP = 0.5L;
		constexpr long double X_TOL = 1e-4L;
		constexpr long double F_TOL = 1e-6L;
		size_t n = initial.size();
		vector<vector<long double>> simplex(n + 1, initial);
		for (size_t i = 0; i < n; ++i)
			simplex[i + 1][i] += STEP;
		vector<long double> values;
		for (auto const& point : simplex)
			values.push_back(objective(counts, point, fixedTheta));

		for (size_t iteration = 0; iteration < maxIterations; ++iteration) {
			vector<size_t> order(n + 1);
			for (size_t i = 0; i <= n; ++i) order[i] = i;
			std::sort(
				order.begin(), order.end(),
				[&](size_t i, size_t j) {
					return values[i] < values[j];
				}
			);
			vector<vector<long double>> sortedSimplex;
			vector<long double> sortedValues;
			for (size_t index : order) {
				sortedSimplex.push_back(simplex[index]);
				sortedValues.push_back(values[index]);
			}
			simplex.swap(sortedSimplex);
			values.swap(sortedValues);

			long double vertexSpan = 0;
			long double valueSpan = 0;
			for (size_t i = 1; i <= n; ++i) {
				valueSpan = std::max(
					valueSpan, std::abs(values[i] - values[0])
				);
				for (size_t j = 0; j < n; ++j)
					vertexSpan = std::max(
						vertexSpan,
						std::abs(simplex[i][j] - simplex[0][j])
					);
			}
			if (vertexSpan <= X_TOL && valueSpan <= F_TOL) break;

			vector<long double> centroid(n, 0);
			for (size_t i = 0; i < n; ++i)
				for (size_t j = 0; j < n; ++j)
					centroid[j] += simplex[i][j] / n;
			auto combine = [&](
				vector<long double> const& from,
				vector<long double> const& toward,
				long double scale
			) {
				vector<long double> result(n);
				for (size_t j = 0; j < n; ++j)
					result[j] =
						from[j] + scale * (toward[j] - from[j]);
				return result;
			};

			vector<long double> reflected =
				combine(centroid, simplex[n], -1);
			long double reflectedValue =
				objective(counts, reflected, fixedTheta);
			if (
				values[0] <= reflectedValue &&
				reflectedValue < values[n - 1]
			) {
				simplex[n] = std::move(reflected);
				values[n] = reflectedValue;
				continue;
			}
			if (reflectedValue < values[0]) {
				vector<long double> expanded =
					combine(centroid, reflected, 2);
				long double expandedValue =
					objective(counts, expanded, fixedTheta);
				if (expandedValue < reflectedValue) {
					simplex[n] = std::move(expanded);
					values[n] = expandedValue;
				}
				else {
					simplex[n] = std::move(reflected);
					values[n] = reflectedValue;
				}
				continue;
			}

			vector<long double> contracted =
				combine(centroid, simplex[n], 0.5L);
			long double contractedValue =
				objective(counts, contracted, fixedTheta);
			if (contractedValue < values[n]) {
				simplex[n] = std::move(contracted);
				values[n] = contractedValue;
				continue;
			}

			for (size_t i = 1; i <= n; ++i) {
				simplex[i] =
					combine(simplex[0], simplex[i], 0.5L);
				values[i] =
					objective(counts, simplex[i], fixedTheta);
			}
		}

		size_t best = 0;
		for (size_t i = 1; i <= n; ++i)
			if (values[i] < values[best]) best = i;
		return {simplex[best], values[best]};
	}

public:
	static long double logLikelihood(
		PatternCounts const& counts,
		long double s1,
		long double s2,
		long double theta
	) noexcept {
		if (!(s1 > 0) || !(s2 > s1) || !(theta > 0))
			return -std::numeric_limits<long double>::infinity();
		auto probabilities =
			branch_length::jc69_msc::triplet(s1, s2, theta);
		long double result = 0;
		for (size_t i = 0; i < counts.size(); ++i) {
			if (!(probabilities[i] > 0))
				return -std::numeric_limits<long double>::infinity();
			if (counts[i] != 0)
				result += counts[i] * std::log(probabilities[i]);
		}
		return result;
	}

	static FitResult fit(
		PatternCounts const& counts,
		std::optional<long double> fixedTheta = std::nullopt
	) noexcept {
		long double total = 0;
		for (long double count : counts) total += count;
		if (!(total > 0) || !std::isfinite(total)) return {};
		if (
			fixedTheta &&
			(!(*fixedTheta > 0) || !std::isfinite(*fixedTheta))
		) return {};

		vector<long double> base =
			initialParameters(counts, fixedTheta);
		constexpr array<long double, 5> THETA_SCALES = {{
			0.01L, 0.1L, 1, 10, 100
		}};
		vector<long double> bestPoint;
		long double bestValue =
			std::numeric_limits<long double>::infinity();
		size_t nStarts = fixedTheta ? 1 : THETA_SCALES.size();
		for (size_t iStart = 0; iStart < nStarts; ++iStart) {
			vector<long double> start = base;
			if (!fixedTheta)
				start.back() += std::log(THETA_SCALES[iStart]);
			auto [point, value] =
				nelderMead(counts, start, fixedTheta);
			if (value < bestValue) {
				bestValue = value;
				bestPoint = std::move(point);
			}
		}
		if (bestPoint.empty()) return {};

		array<long double, 2> ages{};
		long double theta = 0;
		if (!unpack(bestPoint, fixedTheta, ages, theta)) return {};
		if (!(ages[0] > 0) || !(ages[1] > ages[0])) return {};
		return {true, -bestValue, theta, ages};
	}
};

template<typename Color> class PooledTripletTraversal {
public:
	using Tree = common::AnnotatedBinaryTree;
	using Node = Tree::Node;
	using PatternCounts = typename Color::TripletPatternCounts;
	using ThreadPool =
		thread_pool::ThreadPool<
			PatternCounts, thread_pool::SimpleScheduler
		>;

private:
	Color& color;
	ThreadPool& threadPool;
	Tree& tree;
	Node* outgroup;
	thread_pool::Instruction<PatternCounts> instruction;
	vector<Node*> resultNodes;

	vector<size_t> leaves(Node* node) const {
		vector<Node*> leaves;
		node->subtreeLeaves(leaves);
		vector<size_t> result;
		for (Node* leaf : leaves)
			result.push_back(leaf->template get<size_t>(Tree::LEAF_ID));
		return result;
	}

	void setColor(vector<size_t> const& taxa, size_t iColor) {
		Color& localColor = color;
		instruction.mapFuncs.emplace_back(
			[taxa, iColor, &localColor](size_t iElement) noexcept {
				for (size_t taxon : taxa)
					localColor.elementSetTaxonColor(
						iElement, taxon, iColor
					);
			}
		);
	}

	void clearColor(vector<size_t> const& taxa, size_t iColor) {
		Color& localColor = color;
		instruction.mapFuncs.emplace_back(
			[taxa, iColor, &localColor](size_t iElement) noexcept {
				for (size_t taxon : taxa)
					localColor.elementClearTaxonColor(
						iElement, taxon, iColor
					);
			}
		);
	}

	void clearSetColor(
		vector<size_t> const& taxa,
		size_t fromColor,
		size_t toColor
	) {
		Color& localColor = color;
		instruction.mapFuncs.emplace_back(
			[
				taxa, fromColor, toColor, &localColor
			](size_t iElement) noexcept {
				for (size_t taxon : taxa) {
					localColor.elementClearTaxonColor(
						iElement, taxon, fromColor
					);
					localColor.elementSetTaxonColor(
						iElement, taxon, toColor
					);
				}
			}
		);
	}

	void compute(
		Node* leaf,
		array<size_t, 3> colorOrder
	) {
		Color& localColor = color;
		instruction.mapFuncs.emplace_back(
			std::function<PatternCounts(size_t)>(
				[colorOrder, &localColor](size_t iElement) noexcept {
					PatternCounts result{};
					localColor.elementAccumulateTripletPatternCounts(
						iElement, colorOrder, result
					);
					return result;
				}
			)
		);
		instruction.reduceFuncs.push_back(
			[](PatternCounts a, PatternCounts b) noexcept {
				for (size_t i = 0; i < a.size(); ++i)
					a[i] += b[i];
				return a;
			}
		);
		instruction.zeros.push_back(PatternCounts{});
		resultNodes.push_back(leaf);
	}

	void traverse(Node* node) {
		if (node->isLeaf()) {
			clearSetColor(leaves(node), 0, 2);
			return;
		}

		vector<size_t> rightTaxa = leaves(node->rightChild());
		traverse(node->rightChild());
		clearSetColor(rightTaxa, 2, 0);
		traverse(node->leftChild());

		Node* left = node->leftChild();
		Node* right = node->rightChild();
		clearSetColor(rightTaxa, 0, 1);
		// The state here is exactly the three parts around either
		// leaf edge: left=2, right=1, root-side complement=0.
		if (!node->isRoot()) {
			if (left->isLeaf()) compute(left, {{2, 1, 0}});
			if (right->isLeaf()) compute(right, {{1, 2, 0}});
		}
		else {
			Node* ingroup = left == outgroup ? right : left;
			size_t ingroupColor = ingroup == left ? 2 : 1;
			size_t outgroupColor = outgroup == left ? 2 : 1;
			vector<size_t> moved =
				leaves(ingroup->rightChild());
			clearSetColor(moved, ingroupColor, 3);
			compute(
				outgroup,
				{{ingroupColor, 3, outgroupColor}}
			);
			clearSetColor(moved, 3, ingroupColor);
		}
		clearSetColor(rightTaxa, 1, 2);
	}

public:
	PooledTripletTraversal(
		Color& color,
		ThreadPool& threadPool,
		Tree& tree,
		Node* outgroup
	) : color(color), threadPool(threadPool), tree(tree),
		outgroup(outgroup) {}

	vector<std::pair<Node*, PatternCounts>> count() {
		tree.makeLeftHeavyByLeafCount();
		vector<size_t> allTaxa = leaves(tree.root());
		setColor(allTaxa, 0);
		traverse(tree.root());
		clearColor(allTaxa, 2);
		vector<PatternCounts> counts = threadPool(instruction);
		if (counts.size() != resultNodes.size())
			throw std::logic_error(
				"Terminal triplet traversal produced mismatched results."
			);
		vector<std::pair<Node*, PatternCounts>> result;
		for (size_t i = 0; i < counts.size(); ++i)
			result.emplace_back(resultNodes[i], counts[i]);
		return result;
	}
};

template<typename Color> class Procedure {
	using Tree = common::AnnotatedBinaryTree;
	using Node = Tree::Node;
	using PatternCounts = typename Color::TripletPatternCounts;
	using Traversal = PooledTripletTraversal<Color>;
	using ThreadPool = typename Traversal::ThreadPool;

	static Node* sibling(Node* node) noexcept {
		Node* parent = node->parent();
		return parent->leftChild() == node ?
			parent->rightChild() : parent->leftChild();
	}

public:
	static void annotate(
		typename Color::SharedConstData const& data,
		Tree& tree,
		size_t nThreads,
		size_t outgroupTaxon,
		int verbose
	) {
		common::LogInfo(verbose).log()
			<< "Annotating rooted terminal branch lengths (MSC+JC69) ..."
			<< std::endl;
		vector<Node*> allLeaves = tree.leaves();
		if (allLeaves.size() < 3)
			throw std::invalid_argument(
				"Terminal branch lengths require at least three taxa."
			);

		Node* outgroup = nullptr;
		for (Node* leaf : allLeaves)
			if (leaf->template get<size_t>(Tree::LEAF_ID) == outgroupTaxon)
				outgroup = leaf;
		if (!outgroup || outgroup->parent() != tree.root())
			throw std::invalid_argument(
				"The requested outgroup is not a child of the rooted tree."
			);
		Node* ingroup = sibling(outgroup);
		if (ingroup->isLeaf())
			throw std::invalid_argument(
				"Rooted triplet estimation requires a nontrivial ingroup."
			);

		std::optional<long double> fixedTheta;
		if (tree.template has<double>("theta")) {
			fixedTheta = tree.template get<double>("theta");
			common::LogInfo(verbose).log()
				<< "Using internal-quartet global theta "
				<< (double)*fixedTheta
				<< " for the two-age terminal fits." << std::endl;
		}
		else {
			common::LogInfo(verbose).log()
				<< "No internal-edge theta is available; jointly fitting "
				<< "theta with the two triplet ages." << std::endl;
		}

		Color color(data, Color::SiteView::ALL_SITES);
		ThreadPool threadPool(nThreads, 0, data.nElements());
		Traversal traversal(color, threadPool, tree, outgroup);
		auto pooledByLeaf = traversal.count();
		vector<Node*> fittedNodes;
		vector<long double> fittedThetas;
		size_t failed = 0;

		for (auto const& [leaf, counts] : pooledByLeaf) {
			size_t focalTaxon =
				leaf->template get<size_t>(Tree::LEAF_ID);
			bool focalIsOutgroup = leaf == outgroup;
			FitResult fit =
				Estimator<PatternCounts>::fit(counts, fixedTheta);
			if (!fit.success) {
				++failed;
				common::LogInfo(-1).log()
					<< "Rooted terminal branch-length optimization failed "
					<< "for taxon "
					<< common::taxonName2ID[focalTaxon] << "."
					<< std::endl;
				continue;
			}

			// For an ordinary leaf, the leaf/sibling split occurs at s1.
			// The outgroup itself is the C lineage in ((A,B),C), so its
			// pendant branch extends from the present to s2.
			long double focalSubstitution = focalIsOutgroup ?
				fit.speciesAges[1] : fit.speciesAges[0];
			leaf->set("length", (double)focalSubstitution);
			leaf->set("theta_used", (double)fit.theta);
			leaf->set(
				"terminal_branch_length_log_likelihood",
				(double)fit.logLikelihood
			);
			fittedNodes.push_back(leaf);
			fittedThetas.push_back(fit.theta);
		}

		if (fittedNodes.empty())
			throw std::runtime_error(
				"Rooted terminal branch-length optimization failed "
				"for every leaf."
			);

		long double globalTheta;
		if (fixedTheta) {
			globalTheta = *fixedTheta;
		}
		else {
			std::sort(fittedThetas.begin(), fittedThetas.end());
			size_t middle = fittedThetas.size() / 2;
			globalTheta = fittedThetas.size() % 2 ?
				fittedThetas[middle] :
				(fittedThetas[middle - 1] + fittedThetas[middle]) / 2;
			tree.set("theta", (double)globalTheta);
			common::LogInfo(0).log()
				<< "Estimated global theta (median of "
				<< fittedThetas.size() << " rooted terminal fits): "
				<< (double)globalTheta << std::endl;
		}
		for (Node* node : fittedNodes) {
			long double substitution = node->template get<double>("length");
			node->set(
				"length_cu",
				(double)(2 * substitution / globalTheta)
			);
		}
		if (failed) {
			common::LogInfo(-1).log()
				<< failed << " of " << allLeaves.size()
				<< " rooted terminal branch-length fits failed."
				<< std::endl;
		}
	}
};

} // namespace terminal_branch_length

#endif
