from neo4j import GraphDatabase

# ============================================================
#                 SCENARIO SWITCH (for path counting)
# ============================================================

SCENARIO = "remote_unlock"   # headlamp / navigation / remote_unlock

if SCENARIO == "headlamp":
    SOURCE = "OBD"
    TARGET = "HL"
    SCENARIO_IMPACT = 2

elif SCENARIO == "navigation":
    SOURCE = "Cellular"
    TARGET = "HL"
    SCENARIO_IMPACT = 2

elif SCENARIO == "remote_unlock":
    SOURCE = "Cellular"      # main entry
    TARGET = "DoorLock"      # target component
    SCENARIO_IMPACT = 3      # higher impact: physical access

else:
    raise ValueError("Unknown SCENARIO")

print("SCENARIO:", SCENARIO, "SOURCE:", SOURCE, "TARGET:", TARGET)


# ============================================================
#  Neo4j connection
# ============================================================

driver = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=("neo4j", "comealongway")
)


# ============================================================
#  TABLE 6: HEADLAMP SCENARIO (FROM PAPER)
# ============================================================

TABLE6_PATHS = [
    {"idx": 1, "threats": ['UN-T18','UN-T7','UN-T24','UN-T7','UN-T24']},
    {"idx": 2, "threats": ['UN-T18','UN-T7','UN-T24','UN-T8','UN-T24']},
    {"idx": 3, "threats": ['UN-T16','UN-T16','UN-T5','UN-T24','UN-T6','UN-T24']},
    {"idx": 4, "threats": ['UN-T18','UN-T8','UN-T25','UN-T7','UN-T25']},
    {"idx": 5, "threats": ['UN-T18','UN-T6','UN-T25','UN-T8','UN-T24']},
    {"idx": 6, "threats": ['UN-T18','UN-T6','UN-T25','UN-T7','UN-T24']},
    {"idx": 7, "threats": ['UN-T18','UN-T5','UN-T24','UN-T5','UN-T25']},
    {"idx": 8, "threats": ['UN-T18','UN-T7','UN-T25','UN-T7','UN-T25']},
    {"idx": 9, "threats": ['UN-T18','UN-T8','UN-T24','UN-T6','UN-T25']},
]

HL_IMPACT = 2.0   # for the HL ECU in the original Table 6


# ============================================================
#  REMOTE UNLOCK SCENARIO THREAT SEQUENCES (Scenario 2)
# ============================================================

# These correspond to:
#   ru0: Cellular -> TCU -> GW -> CAN_Body -> DLfc
#   ru1: BLE      -> TCU -> GW -> CAN_Body -> DLfc
REMOTE_UNLOCK_PATHS = [
    {
        "idx": 1,
        "label": "ru0",
        "threats": ['UN-T30', 'UN-T6', 'UN-T5', 'UN-T24'],
    },
    {
        "idx": 2,
        "label": "ru1",
        "threats": ['UN-T31', 'UN-T6', 'UN-T5', 'UN-T24'],
    }
]


# ============================================================
#  LOAD GRAPH
# ============================================================

def load_graph(tx):
    states = {}
    for r in tx.run("""
        MATCH (s:State)
        RETURN elementId(s) AS id,
               s.component  AS component
    """):
        states[r["id"]] = {"component": r["component"]}

    edges = []
    for r in tx.run("""
        MATCH (s:State)-[:LEADS_TO|TRANSITION]->(t:State)
        RETURN elementId(s) AS sid, elementId(t) AS tid
    """):
        edges.append((r["sid"], r["tid"]))

    return states, edges


# ============================================================
#  PATH PRINTING HELPER
# ============================================================

def format_state_path(states, node_ids):
    return [states[n]['component'] for n in node_ids]


# ============================================================
#  BFS
# ============================================================

def bfs_paths(states, edges, source_component, target_component, max_depth=12):
    from collections import defaultdict, deque

    adj = defaultdict(list)
    for s, t in edges:
        adj[s].append(t)

    starts = [sid for sid, v in states.items() if v["component"] == source_component]
    goals  = {sid for sid, v in states.items() if v["component"] == target_component}

    q = deque((s, [s]) for s in starts)
    paths = []

    while q:
        node, path = q.popleft()
        if len(path) > max_depth:
            continue

        if node in goals and len(path) > 1:
            paths.append(path)

        for nbr in adj[node]:
            if nbr in path:
                continue
            q.append((nbr, path + [nbr]))

    return paths


# ============================================================
#  DFS
# ============================================================

def dfs_paths(states, edges, source_component, target_component, max_depth=12):
    from collections import defaultdict

    adj = defaultdict(list)
    for s, t in edges:
        adj[s].append(t)

    starts = [sid for sid, v in states.items() if v["component"] == source_component]
    goals  = {sid for sid, v in states.items() if v["component"] == target_component}

    stack = [(s, [s]) for s in starts]
    paths = []

    while stack:
        node, path = stack.pop()
        if len(path) > max_depth:
            continue

        if node in goals and len(path) > 1:
            paths.append(path)

        for nbr in adj[node]:
            if nbr in path:
                continue
            stack.append((nbr, path + [nbr]))

    return paths


# ============================================================
#  VALIDATION FOR SCENARIO 2
# ============================================================

def validate_remote_unlock_paths(states, unique_node_paths):
    expected = {
        "ru0": ("Cellular", "TCU", "GW", "CAN_Body", "DoorLock"),
        "ru1": ("BLE",      "TCU", "GW", "CAN_Body", "DoorLock"),
    }

    comp_paths = {
        tuple(format_state_path(states, list(p)))
        for p in unique_node_paths
    }

    print("=== VALIDATION: REMOTE_UNLOCK ===")
    for pid, epath in expected.items():
        if epath in comp_paths:
            print(f"[OK] {pid} found: {epath}")
        else:
            print(f"[FAIL] {pid} NOT found: {epath}")
    print()


# ============================================================
#  LOAD THREAT FEASIBILITY
# ============================================================

def get_threat_feasibility(tx, threat_ids):
    feas = {}
    for tid in threat_ids:
        r = tx.run("""
            MATCH (t:Threat {id:$tid})
            RETURN t.feasibility AS f
        """, tid=tid).single()
        if r and r["f"] is not None:
            feas[tid] = r["f"]
    return feas


# ============================================================
#  NORMALIZATION
# ============================================================

def normalized_feas(sumF):
    if sumF >= 23: return 0.0
    if sumF >= 21: return 1.0
    if sumF >= 19: return 1.5
    return 2.0


# ============================================================
#  GENERIC RISK TABLE COMPUTATION
# ============================================================

def compute_risk_table(rows, feas_map, impact_rating):
    """
    rows: list of {"idx":..., "threats":[...]}
    feas_map: {'UN-Txx': [ET,SE,KoIC,WoO,EQ]}
    impact_rating: scalar
    """
    results = []

    for row in rows:
        idx = row["idx"]
        threat_list = row["threats"]
        path_length = len(threat_list) - 1

        feas_vectors = []
        missing = []
        for t in threat_list:
            if t in feas_map:
                feas_vectors.append(feas_map[t])
            else:
                missing.append(t)

        if missing:
            print(f"[WARN] Missing feasibility for threats in row {idx}: {missing}")
        if not feas_vectors:
            continue

        FM = [max(col) for col in zip(*feas_vectors)]
        sumF = sum(FM)
        norm = normalized_feas(sumF)
        risk = norm * impact_rating

        results.append({
            "idx": idx,
            "threats": threat_list,
            "length": path_length,
            "FM": FM,
            "sumF": sumF,
            "norm": norm,
            "risk": risk
        })

    return results


# ============================================================
#  MAIN
# ============================================================

with driver.session() as session:

    # --------- PART A: PATH DISCOVERY & VALIDATION ---------
    states, edges = session.execute_read(load_graph)

    # For remote_unlock, we have two sources: Cellular & BLE
    if SCENARIO == "remote_unlock":
        source_components = ["Cellular", "BLE"]
    else:
        source_components = [SOURCE]

    bfs_res = []
    dfs_res = []
    for src in source_components:
        bfs_res.extend(bfs_paths(states, edges, src, TARGET))
        dfs_res.extend(dfs_paths(states, edges, src, TARGET))

    all_paths = bfs_res + dfs_res
    unique_paths = {tuple(p) for p in all_paths}

    print(f"\n>>> UNIQUE ATTACK PATHS GENERATED (BFS + DFS): {len(unique_paths)}\n")

    print("Some example paths:\n")
    for i, path in enumerate(list(unique_paths)[:10], 1):
        print(f"Path {i}: {format_state_path(states, list(path))}")
    print()

    if SCENARIO == "remote_unlock":
        validate_remote_unlock_paths(states, unique_paths)

    # --------- PART B: TABLE 6 + REMOTE_UNLOCK TABLE ---------

    # Collect all threat IDs needed for both tables
    all_threat_ids = sorted({
        t
        for row in TABLE6_PATHS
        for t in row["threats"]
    } | {
        t
        for row in REMOTE_UNLOCK_PATHS
        for t in row["threats"]
    })

    feas_map = session.execute_read(get_threat_feasibility, all_threat_ids)

    # Impact for HL from DB if available
    hl_record = session.run("""
        MATCH (hl:ECU {name:'HL'})
        RETURN hl.impact_rating AS impact
    """).single()
    headlamp_impact = hl_record["impact"] if hl_record and hl_record["impact"] else HL_IMPACT

    # Original Table 6 (HL scenario, from paper)
    table6_results = compute_risk_table(TABLE6_PATHS, feas_map, headlamp_impact)

    # New risk table for Remote Unlock (Scenario 2)
    remote_unlock_results = compute_risk_table(
        REMOTE_UNLOCK_PATHS,
        feas_map,
        SCENARIO_IMPACT  # 3 for DoorLock
    )

# ============================================================
#  PRINT: TABLE 6 (HEADLAMP)
# ============================================================

print("\n=== TABLE 6 — ORIGINAL HEADLAMP SCENARIO (Validation) ===\n")

for r in sorted(table6_results, key=lambda x: x["idx"]):
    print(f"Row {r['idx']}:")
    print(f"  Threats:               {r['threats']}")
    print(f"  PathLength:            {r['length']}")
    print(f"  FM:                    {r['FM']}")
    print(f"  Feasibility Sum:       {r['sumF']}")
    print(f"  Normalized Feasibility:{r['norm']}")
    print(f"  Risk:                  {r['risk']}")
    print("-"*75)

# ============================================================
#  PRINT: TABLE 6b (REMOTE UNLOCK)
# ============================================================

print("\n=== TABLE 6b_remote — REMOTE UNLOCK SCENARIO (New Risk Table) ===\n")

# build idx -> label map for nicer printing
label_map = {row["idx"]: row["label"] for row in REMOTE_UNLOCK_PATHS}

for r in sorted(remote_unlock_results, key=lambda x: x["idx"]):
    label = label_map.get(r["idx"], f"ru{r['idx']}")
    print(f"Row {r['idx']} ({label}):")
    print(f"  Threats:               {r['threats']}")
    print(f"  PathLength:            {r['length']}")
    print(f"  FM:                    {r['FM']}")
    print(f"  Feasibility Sum:       {r['sumF']}")
    print(f"  Normalized Feasibility:{r['norm']}")
    print(f"  Risk:                  {r['risk']}")
    print("-"*75)
