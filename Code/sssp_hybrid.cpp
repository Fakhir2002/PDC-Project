// Compile: mpic++ -fopenmp -std=c++17 sssp_mpi.cpp -o sssp_mpi
// Run: mpirun -np 2 --hostfile hosts ./sssp_mpi 0 5



#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <vector>
#include <queue>
#include <fstream>
#include <sstream>
#include <limits>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <chrono>
#include <tuple>

using namespace std;
using namespace chrono;

const int INF = numeric_limits<int>::max();
typedef pair<int, int> pii;

// Load local partition file
void load_local_partition(const string& filename, vector<vector<pii>>& adj,
                          unordered_set<int>& local_nodes, unordered_set<int>& boundary_nodes,
                          int& max_node, vector<tuple<int, int, int>>& edges) {
    ifstream infile(filename);
    int u, v, w;
    max_node = 0;
    while (infile >> u >> v >> w) {
        max_node = max(max_node, max(u, v));
        adj.resize(max_node + 1);
        adj[u].push_back({v, w});
        edges.emplace_back(u, v, w);
        local_nodes.insert(u);
        if (!local_nodes.count(v)) boundary_nodes.insert(v);
    }
}

// Serial Dijkstra for initial SSSP on rank 0
void recompute_sssp(vector<vector<pii>>& adj, vector<int>& dist, vector<int>& parent, int source) {
    int n = adj.size();
    dist.assign(n, INF);
    parent.assign(n, -1);
    dist[source] = 0;

    priority_queue<pii, vector<pii>, greater<pii>> pq;
    pq.push({0, source});

    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d > dist[u]) continue;
        for (auto [v, w] : adj[u]) {
            if (dist[v] > dist[u] + w) {
                dist[v] = dist[u] + w;
                parent[v] = u;
                pq.push({dist[v], v});
            }
        }
    }
}

// Broadcast a vector<int> to all ranks
void broadcast_vector(vector<int>& v, int root) {
    int size = v.size();
    MPI_Bcast(&size, 1, MPI_INT, root, MPI_COMM_WORLD);
    v.resize(size);
    MPI_Bcast(v.data(), size, MPI_INT, root, MPI_COMM_WORLD);
}

// Simulate edge updates on rank 0
void simulate_updates(const vector<tuple<int, int, int>>& edges,
                      vector<tuple<int, int, int>>& deletions,
                      vector<tuple<int, int, int>>& insertions,
                      int num_del, int num_ins, int max_node) {
    srand(42);
    unordered_set<int> used;
    while (deletions.size() < num_del) {
        int i = rand() % edges.size();
        if (used.count(i)) continue;
        used.insert(i);
        deletions.push_back(edges[i]);
    }

    unordered_set<string> existing;
    for (auto [u, v, w] : edges)
        existing.insert(to_string(u) + "," + to_string(v));

    while (insertions.size() < num_ins) {
        int u = rand() % max_node;
        int v = rand() % max_node;
        if (u == v) continue;
        string key = to_string(u) + "," + to_string(v);
        if (existing.count(key)) continue;
        int w = rand() % 10 + 1;
        insertions.push_back({u, v, w});
        existing.insert(key);
    }
}

void process_deletions(vector<vector<pii>>& adj, vector<int>& dist, vector<int>& parent,
                       const vector<tuple<int, int, int>>& deletions,
                       vector<bool>& affected, vector<bool>& affected_del) {
    for (auto [u, v, w] : deletions) {
        adj[u].erase(remove_if(adj[u].begin(), adj[u].end(),
            [v](pii edge) { return edge.first == v; }), adj[u].end());
        if (parent[v] == u) {
            queue<int> q;
            q.push(v);
            affected[v] = true;
            affected_del[v] = true;
            dist[v] = INF;
            parent[v] = -1;
            while (!q.empty()) {
                int u = q.front(); q.pop();
                for (auto [v2, w2] : adj[u]) {
                    if (parent[v2] == u && !affected_del[v2]) {
                        affected_del[v2] = true;
                        affected[v2] = true;
                        dist[v2] = INF;
                        parent[v2] = -1;
                        q.push(v2);
                    }
                }
            }
        }
    }
}

void process_insertions(vector<vector<pii>>& adj,
                        const vector<tuple<int, int, int>>& insertions,
                        vector<int>& dist, vector<int>& parent,
                        vector<bool>& affected, const vector<bool>& affected_del) {
    for (auto [u, v, w] : insertions) {
        if (affected_del[v]) continue;
        if (dist[u] != INF && dist[u] + w < dist[v]) {
            adj[u].push_back({v, w});
            dist[v] = dist[u] + w;
            parent[v] = u;
            affected[v] = true;
        }
    }
}

void exchange_ghost_distances(unordered_set<int>& boundary_nodes, vector<int>& dist, int rank) {
    vector<int> send_buf, recv_buf;
    for (int node : boundary_nodes) {
        send_buf.push_back(node);
        send_buf.push_back(dist[node]);
    }

    int send_size = send_buf.size(), recv_size = 0;
    MPI_Sendrecv(&send_size, 1, MPI_INT, 1 - rank, 0,
                 &recv_size, 1, MPI_INT, 1 - rank, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    recv_buf.resize(recv_size);

    MPI_Sendrecv(send_buf.data(), send_size, MPI_INT, 1 - rank, 1,
                 recv_buf.data(), recv_size, MPI_INT, 1 - rank, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    for (int i = 0; i < recv_size; i += 2) {
        int node = recv_buf[i], d = recv_buf[i + 1];
        if (dist[node] > d) dist[node] = d;
    }
}

void async_update(vector<vector<pii>>& adj, vector<int>& dist, vector<int>& parent,
                  vector<bool>& affected, int async_depth, const unordered_set<int>& local_nodes, int rank) {
    int n = dist.size();
    bool changed = true;
    while (changed) {
        changed = false;

        #pragma omp parallel for schedule(dynamic)
        for (int u = 0; u < n; ++u) {
            if (!affected[u] || dist[u] == INF || !local_nodes.count(u)) continue;

            queue<pair<int, int>> q;
            q.push({u, 0});
            affected[u] = false;

            while (!q.empty()) {
                auto [curr, depth] = q.front(); q.pop();
                for (auto [v, w] : adj[curr]) {
                    if (dist[v] > dist[curr] + w) {
                        dist[v] = dist[curr] + w;
                        parent[v] = curr;
                        affected[v] = true;
                        changed = true;
                        if (depth + 1 <= async_depth)
                            q.push({v, depth + 1});
                    }
                }
            }
        }

        exchange_ghost_distances(const_cast<unordered_set<int>&>(local_nodes), dist, rank);

        int local_changed = changed ? 1 : 0, global_changed = 0;
        MPI_Allreduce(&local_changed, &global_changed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        changed = global_changed;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc < 3) {
        if (rank == 0) cerr << "Usage: ./sssp_mpi_openmp <source_node> <async_depth>\n";
        MPI_Finalize();
        return 1;
    }

    int source = atoi(argv[1]);
    int async_depth = atoi(argv[2]);

    string file = "part" + to_string(rank) + ".txt";

    vector<vector<pii>> adj;
    unordered_set<int> local_nodes, boundary_nodes;
    vector<tuple<int, int, int>> edges;
    int max_node = 0;

    load_local_partition(file, adj, local_nodes, boundary_nodes, max_node, edges);
    vector<int> dist, parent;
int global_max_node = 0;

// Only rank 0 loads the full graph and computes SSSP
if (rank == 0) {
    ifstream infile("amazon400.txt");
    int u, v, w;
    vector<tuple<int, int, int>> full_edges;
    while (infile >> u >> v >> w) {
        full_edges.emplace_back(u, v, w);
        global_max_node = max(global_max_node, max(u, v));
    }

    dist.resize(global_max_node + 1, INF);
    parent.resize(global_max_node + 1, -1);
    vector<vector<pii>> full_adj(global_max_node + 1);
    for (auto [u, v, w] : full_edges)
        full_adj[u].push_back({v, w});

    auto start = high_resolution_clock::now();
    recompute_sssp(full_adj, dist, parent, source);
    auto end = high_resolution_clock::now();
    cout << "Initial SSSP completed in "
         << duration_cast<milliseconds>(end - start).count() << " ms\n";
}

// Broadcast global size first
MPI_Bcast(&global_max_node, 1, MPI_INT, 0, MPI_COMM_WORLD);
if (rank != 0) {
    dist.resize(global_max_node + 1);
    parent.resize(global_max_node + 1);
}

// Broadcast the SSSP results to all
MPI_Bcast(dist.data(), global_max_node + 1, MPI_INT, 0, MPI_COMM_WORLD);
MPI_Bcast(parent.data(), global_max_node + 1, MPI_INT, 0, MPI_COMM_WORLD);


    // Get update counts from user
    int num_ins = 0, num_del = 0;
    vector<tuple<int, int, int>> deletions, insertions;

    if (rank == 0) {
        cout << "Enter number of deletions: ";
        cin >> num_del;
        cout << "Enter number of insertions: ";
        cin >> num_ins;
        simulate_updates(edges, deletions, insertions, num_del, num_ins, max_node);
    }

    // Share updates
    MPI_Bcast(&num_del, 1, MPI_INT, 0, MPI_COMM_WORLD);
    deletions.resize(num_del);
    MPI_Bcast(deletions.data(), num_del * sizeof(tuple<int, int, int>), MPI_BYTE, 0, MPI_COMM_WORLD);

    MPI_Bcast(&num_ins, 1, MPI_INT, 0, MPI_COMM_WORLD);
    insertions.resize(num_ins);
    MPI_Bcast(insertions.data(), num_ins * sizeof(tuple<int, int, int>), MPI_BYTE, 0, MPI_COMM_WORLD);

    vector<bool> affected(max_node + 1, false), affected_del(max_node + 1, false);

    auto start = high_resolution_clock::now();
    process_deletions(adj, dist, parent, deletions, affected, affected_del);
    process_insertions(adj, insertions, dist, parent, affected, affected_del);
    async_update(adj, dist, parent, affected, async_depth, local_nodes, rank);
    auto end = high_resolution_clock::now();

    cout << "Rank " << rank << ": dynamic update finished in "
         << duration_cast<milliseconds>(end - start).count() << " ms\n";

    MPI_Finalize();
    return 0;
}
