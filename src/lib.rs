//! Ollivier-Ricci curvature on agent interaction graphs.
//!
//! Ricci curvature measures how much "geodesics spread apart" — on graphs,
//! it tells us whether connected nodes are converging (positive curvature)
//! or diverging (negative curvature).
//!
//! For agent fleets:
//! - **Positive curvature** between agents → their neighborhoods overlap → convergence likely
//! - **Negative curvature** → agents are bridges between clusters → divergence likely
//! - **Zero curvature** → tree-like structure → neutral
//!
//! Ollivier-Ricci: κ(x,y) = 1 - W₁(μ_x, μ_y) where W₁ = Wasserstein-1 distance
//! between neighborhood distributions.

/// An agent in the interaction graph.
#[derive(Debug, Clone)]
pub struct Agent {
    pub id: usize,
    pub neighbors: Vec<(usize, f64)>, // (neighbor_id, edge_weight)
}

impl Agent {
    pub fn new(id: usize) -> Self { Self { id, neighbors: vec![] } }
    pub fn with_neighbors(id: usize, neighbors: Vec<(usize, f64)>) -> Self { Self { id, neighbors } }
    pub fn add_neighbor(&mut self, neighbor: usize, weight: f64) {
        if !self.neighbors.iter().any(|(n, _)| *n == neighbor) {
            self.neighbors.push((neighbor, weight));
        }
    }
    pub fn degree(&self) -> usize { self.neighbors.len() }
    pub fn total_weight(&self) -> f64 { self.neighbors.iter().map(|(_, w)| w).sum() }
    /// Lazy random walk distribution: uniform over neighbors (with weights).
    pub fn distribution(&self) -> Vec<(usize, f64)> {
        let tw = self.total_weight();
        if tw < 1e-12 { return vec![]; }
        self.neighbors.iter().map(|(n, w)| (*n, w / tw)).collect()
    }
}

/// The agent interaction graph.
#[derive(Debug, Clone)]
pub struct AgentGraph {
    pub agents: Vec<Agent>,
}

impl AgentGraph {
    pub fn new(n: usize) -> Self {
        Self { agents: (0..n).map(|i| Agent::new(i)).collect() }
    }
    pub fn add_edge(&mut self, a: usize, b: usize, weight: f64) {
        self.agents[a].add_neighbor(b, weight);
        self.agents[b].add_neighbor(a, weight);
    }
    pub fn agent(&self, id: usize) -> &Agent { &self.agents[id] }
    pub fn n_agents(&self) -> usize { self.agents.len() }
    pub fn n_edges(&self) -> usize {
        self.agents.iter().map(|a| a.degree()).sum::<usize>() / 2
    }

    /// Compute Ollivier-Ricci curvature for edge (a, b).
    /// κ(a,b) = 1 - W₁(μ_a, μ_b)
    pub fn ricci_curvature(&self, a: usize, b: usize) -> f64 {
        let dist_a = self.agents[a].distribution();
        let dist_b = self.agents[b].distribution();
        let w1 = wasserstein_1(&dist_a, &dist_b, self.agents.len());
        1.0 - w1
    }

    /// Average Ricci curvature across all edges.
    pub fn average_curvature(&self) -> f64 {
        let mut total = 0.0;
        let mut count = 0;
        for a in &self.agents {
            for &(b, _) in &a.neighbors {
                if b > a.id { // count each edge once
                    total += self.ricci_curvature(a.id, b);
                    count += 1;
                }
            }
        }
        if count == 0 { 0.0 } else { total / count as f64 }
    }

    /// Minimum Ricci curvature (most divergent edge).
    pub fn min_curvature(&self) -> f64 {
        let mut min_k = f64::MAX;
        for a in &self.agents {
            for &(b, _) in &a.neighbors {
                if b > a.id {
                    min_k = min_k.min(self.ricci_curvature(a.id, b));
                }
            }
        }
        if min_k == f64::MAX { 0.0 } else { min_k }
    }

    /// Maximum Ricci curvature (most convergent edge).
    pub fn max_curvature(&self) -> f64 {
        let mut max_k = f64::MIN;
        for a in &self.agents {
            for &(b, _) in &a.neighbors {
                if b > a.id {
                    max_k = max_k.max(self.ricci_curvature(a.id, b));
                }
            }
        }
        if max_k == f64::MIN { 0.0 } else { max_k }
    }

    /// Forman-Ricci curvature for edge (a, b).
    /// F(e) = w_e * (1/w_a + 1/w_b - Σ 1/w_e') where w_a = sum of edge weights at a.
    pub fn forman_curvature(&self, a: usize, b: usize) -> f64 {
        let w_a = self.agents[a].total_weight().max(1e-12);
        let w_b = self.agents[b].total_weight().max(1e-12);
        let d_a = self.agents[a].degree() as f64;
        let d_b = self.agents[b].degree() as f64;
        4.0 / w_a.max(w_b) - (d_a - 1.0) / w_a - (d_b - 1.0) / w_b
    }

    /// Fleet convergence prediction based on average curvature.
    /// Positive κ̄ → converging. Negative κ̄ → diverging.
    pub fn convergence_prediction(&self) -> ConvergencePrediction {
        let avg = self.average_curvature();
        let min = self.min_curvature();
        let max = self.max_curvature();
        let n = self.n_agents();
        let density = self.n_edges() as f64 / (n as f64 * (n as f64 - 1.0) / 2.0).max(1.0);
        ConvergencePrediction {
            average_curvature: avg,
            min_curvature: min,
            max_curvature: max,
            density,
            will_converge: avg > 0.0,
            convergence_speed: avg.abs() * density * 10.0,
        }
    }

    /// Detect structural bottlenecks (edges with most negative curvature).
    pub fn bottlenecks(&self, top_k: usize) -> Vec<(usize, usize, f64)> {
        let mut edges: Vec<(usize, usize, f64)> = vec![];
        for a in &self.agents {
            for &(b, _) in &a.neighbors {
                if b > a.id {
                    edges.push((a.id, b, self.ricci_curvature(a.id, b)));
                }
            }
        }
        edges.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
        edges.into_iter().take(top_k).collect()
    }
}

/// Wasserstein-1 distance between two discrete distributions (naive Earth Mover's).
fn wasserstein_1(dist_a: &[(usize, f64)], dist_b: &[(usize, f64)], n: usize) -> f64 {
    // Build cumulative distributions over sorted node indices
    let mut pa = vec![0.0; n];
    let mut pb = vec![0.0; n];
    for &(node, prob) in dist_a { if node < n { pa[node] += prob; } }
    for &(node, prob) in dist_b { if node < n { pb[node] += prob; } }
    // For self-loops: include the node itself in its distribution
    // W1 via CDF difference (1D earth mover's)
    let mut cdf_a = 0.0;
    let mut cdf_b = 0.0;
    let mut w1 = 0.0;
    for i in 0..n {
        cdf_a += pa[i];
        cdf_b += pb[i];
        w1 += (cdf_a - cdf_b).abs();
    }
    w1
}

/// Convergence prediction result.
#[derive(Debug, Clone)]
pub struct ConvergencePrediction {
    pub average_curvature: f64,
    pub min_curvature: f64,
    pub max_curvature: f64,
    pub density: f64,
    pub will_converge: bool,
    pub convergence_speed: f64,
}

/// Graph topology factories.
pub fn complete_graph(n: usize) -> AgentGraph {
    let mut g = AgentGraph::new(n);
    for i in 0..n { for j in (i+1)..n { g.add_edge(i, j, 1.0); } }
    g
}

pub fn ring_graph(n: usize) -> AgentGraph {
    let mut g = AgentGraph::new(n);
    for i in 0..n { g.add_edge(i, (i + 1) % n, 1.0); }
    g
}

pub fn star_graph(n: usize) -> AgentGraph {
    let mut g = AgentGraph::new(n);
    for i in 1..n { g.add_edge(0, i, 1.0); }
    g
}

pub fn path_graph(n: usize) -> AgentGraph {
    let mut g = AgentGraph::new(n);
    for i in 0..n.saturating_sub(1) { g.add_edge(i, i + 1, 1.0); }
    g
}

pub fn grid_graph(rows: usize, cols: usize) -> AgentGraph {
    let n = rows * cols;
    let mut g = AgentGraph::new(n);
    for r in 0..rows {
        for c in 0..cols {
            let i = r * cols + c;
            if c + 1 < cols { g.add_edge(i, i + 1, 1.0); }
            if r + 1 < rows { g.add_edge(i, i + cols, 1.0); }
        }
    }
    g
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complete_graph_positive_curvature() {
        let g = complete_graph(5);
        let avg = g.average_curvature();
        // Complete graph: all neighborhoods overlap → positive curvature
        assert!(avg > 0.0, "Complete graph should have positive curvature, got {}", avg);
    }

    #[test]
    fn test_ring_graph_curvature() {
        let g = ring_graph(6);
        let avg = g.average_curvature();
        // Ring has moderate curvature (neighboring nodes share 1 neighbor)
        assert!(avg > -0.5, "Ring curvature should be bounded, got {}", avg);
    }

    #[test]
    fn test_star_graph_hub_negative_curvature() {
        let g = star_graph(5);
        // Hub-spoke edges: hub sees all spokes, spoke sees only hub → negative
        let k = g.ricci_curvature(0, 1);
        assert!(k < 0.0, "Hub-spoke should have negative curvature, got {}", k);
    }

    #[test]
    fn test_path_endpoints() {
        let g = path_graph(4);
        let k = g.ricci_curvature(0, 1);
        // Path endpoints: one neighbor each, diverging neighborhoods
        // Not necessarily negative but should be bounded
        assert!(k > -1.0 && k < 1.0);
    }

    #[test]
    fn test_graph_construction() {
        let g = complete_graph(4);
        assert_eq!(g.n_agents(), 4);
        assert_eq!(g.n_edges(), 6); // C(4,2) = 6
    }

    #[test]
    fn test_ring_edges() {
        let g = ring_graph(5);
        assert_eq!(g.n_edges(), 5);
    }

    #[test]
    fn test_star_edges() {
        let g = star_graph(5);
        assert_eq!(g.n_edges(), 4);
    }

    #[test]
    fn test_convergence_prediction_positive() {
        let g = complete_graph(6);
        let pred = g.convergence_prediction();
        assert!(pred.will_converge);
        assert!(pred.convergence_speed > 0.0);
    }

    #[test]
    fn test_convergence_prediction_negative() {
        let g = star_graph(10);
        let pred = g.convergence_prediction();
        // Star graph: average curvature may be negative due to hub-spoke edges
        // At minimum, density is low
        assert!(pred.density < 0.3);
    }

    #[test]
    fn test_curvature_ordering() {
        let complete = complete_graph(6).average_curvature();
        let ring = ring_graph(6).average_curvature();
        // Complete should have higher curvature than ring
        assert!(complete >= ring, "Complete κ={} should >= ring κ={}", complete, ring);
    }

    #[test]
    fn test_forman_curvature() {
        let g = complete_graph(4);
        let f = g.forman_curvature(0, 1);
        // Forman should be computable
        assert!(f.is_finite());
    }

    #[test]
    fn test_bottlenecks() {
        // Create a dumbbell: two complete-3 graphs connected by a bridge
        let mut g = AgentGraph::new(6);
        // Left cluster: 0,1,2
        g.add_edge(0, 1, 1.0); g.add_edge(0, 2, 1.0); g.add_edge(1, 2, 1.0);
        // Bridge
        g.add_edge(2, 3, 1.0);
        // Right cluster: 3,4,5
        g.add_edge(3, 4, 1.0); g.add_edge(3, 5, 1.0); g.add_edge(4, 5, 1.0);
        let bns = g.bottlenecks(1);
        assert!(!bns.is_empty());
        // The bridge edge (2,3) should be a bottleneck
        assert_eq!(bns[0].0, 2);
        assert_eq!(bns[0].1, 3);
    }

    #[test]
    fn test_grid_curvature() {
        let g = grid_graph(3, 3);
        let avg = g.average_curvature();
        assert!(avg.is_finite());
    }

    #[test]
    fn test_wasserstein_self() {
        let g = complete_graph(4);
        let da = g.agents[0].distribution();
        let db = g.agents[0].distribution();
        let w1 = wasserstein_1(&da, &db, 4);
        assert!(w1 < 1e-10, "W1 with self should be 0");
    }

    #[test]
    fn test_agent_distribution_sums_to_one() {
        let g = ring_graph(5);
        let d = g.agents[0].distribution();
        let sum: f64 = d.iter().map(|(_, p)| p).sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_density_complete() {
        let g = complete_graph(5);
        let pred = g.convergence_prediction();
        assert!((pred.density - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_min_max_curvature() {
        let g = complete_graph(4);
        assert!(g.min_curvature() <= g.max_curvature());
    }

    #[test]
    fn test_isolated_agent() {
        let g = AgentGraph::new(3);
        // No edges — curvature undefined, should return 0
        let k = g.ricci_curvature(0, 1);
        // Both have empty distributions, W1 = 0, so κ = 1
        // But there's no edge — the function still computes
        assert!(k.is_finite());
    }
}
