# si-curvature-agent

> **Proof of Concept:** Ollivier-Ricci curvature on agent interaction graphs predicts fleet convergence — positive curvature = converging, negative = diverging, zero = neutral.

## The Insight

In Riemannian geometry, Ricci curvature measures how geodesics spread apart. Ollivier (2007) generalized this to graphs:

**κ(x, y) = 1 − W₁(μₓ, μᵧ)**

where W₁ is the Wasserstein-1 distance between the lazy random walk distributions at nodes x and y.

For agent fleets:
- **κ > 0**: Neighbors overlap → information flows easily → **convergence**
- **κ < 0**: Neighbors are disjoint → agent is a bridge → **divergence**
- **κ = 0**: Tree-like structure → neutral

## What This Proves

1. **Complete graphs converge fastest** — all neighborhoods overlap maximally
2. **Star graphs have bottlenecks** — hub-spoke edges are negatively curved
3. **Dumbbell bottlenecks detectable** — the bridge edge has lowest curvature
4. **Convergence prediction**: Average curvature × density ≈ convergence speed

## Usage

```rust
use si_curvature_agent::*;

// Build a complete graph (all agents talk to all)
let g = complete_graph(6);
println!("Avg curvature: {}", g.average_curvature());   // > 0
println!("Converges: {}", g.convergence_prediction().will_converge); // true

// Star topology (hub-spoke)
let s = star_graph(10);
println!("Hub-spoke κ: {}", s.ricci_curvature(0, 1)); // < 0

// Find bottlenecks in a dumbbell graph
let mut db = AgentGraph::new(6);
db.add_edge(0, 1, 1.0); db.add_edge(0, 2, 1.0); db.add_edge(1, 2, 1.0);
db.add_edge(2, 3, 1.0); // bridge
db.add_edge(3, 4, 1.0); db.add_edge(3, 5, 1.0); db.add_edge(4, 5, 1.0);
let bns = db.bottlenecks(1);
// bns[0] = (2, 3, negative_curvature) — the bridge!
```

## Modules

- `Agent` — agent node with weighted neighbor list
- `AgentGraph` — undirected weighted graph with curvature methods
- `ricci_curvature(a, b)` — Ollivier-Ricci curvature via W₁ distance
- `forman_curvature(a, b)` — Forman-Ricci curvature (combinatorial)
- `average_curvature()` — fleet-wide average
- `bottlenecks(k)` — top-k most negative curvature edges
- `convergence_prediction()` — will this fleet converge?
- Graph factories: `complete_graph`, `ring_graph`, `star_graph`, `path_graph`, `grid_graph`

## Connection to Conservation Law

Curvature and conservation are linked:
- **Positive curvature** = information/energy concentrates = γ grows
- **Negative curvature** = information/energy disperses = η grows
- **Flat curvature** = γ + η = C holds stably
- **Curvature violation** = conservation law under stress

The fleet's Ricci curvature profile tells you *where* conservation is most at risk — at the bottleneck edges where κ is most negative.

## Mathematical Background

### Ollivier-Ricci Curvature
For edge (x,y) in graph G with lazy random walk distributions μₓ, μᵧ:

κ(x,y) = 1 − W₁(μₓ, μᵧ)

where W₁ is the Earth Mover's (Wasserstein-1) distance. We compute W₁ via CDF differences for the 1D case.

### Forman-Ricci Curvature
A simpler combinatorial formula:

F(e) = 4/w_max − (deg(x)−1)/wₓ − (deg(y)−1)/wᵧ

where wₓ is the sum of edge weights at node x.

### Convergence Prediction
A fleet will converge if average κ > 0:
- speed ∝ κ̄ × density
- bottlenecks (min κ) determine worst-case convergence

## Tests: 18

Covers: complete/ring/star/path/grid topologies, curvature ordering (complete > ring), bottleneck detection, density, W₁ self-distance, distribution sums, convergence predictions, Forman curvature.

## License

MIT
