import pickle

JACK_NODES_NUMBER = 195
COPS_NODES_NUMBER = 234


class Node:
    def __init__(self, given_id, edges, neighbours):
        self.id = given_id
        self.edges = edges
        self.neighbours = neighbours


class tmpNode:
    id = 1

    def __init__(self):
        self.id = self.__class__.id
        self.__class__.id += 1
        self.edges = []
        self.neighbours = []

    def add_edge(self, node):
        self.edges.append(node)

    def add_neighbour(self, neighbour):
        self.neighbours.append(neighbour)

    def purify(self):
        return Node(self.id, self.edges, self.neighbours)


class NodeJack(tmpNode):
    pass


class NodeCops(tmpNode):
    pass


def jack_add_edge(node_1, node_2):
    jack_nodes[node_1 - 1].add_edge(jack_nodes[node_2 - 1])
    jack_nodes[node_2 - 1].add_edge(jack_nodes[node_1 - 1])


def cops_add_edge(node_1, node_2):
    cops_nodes[node_1 - 1].add_edge(cops_nodes[node_2 - 1])
    cops_nodes[node_2 - 1].add_edge(cops_nodes[node_1 - 1])


def add_neighbour(node_1, node_2):
    node_1.add_neighbour(node_2)
    node_2.add_neighbour(node_1)


jack_nodes = [NodeJack() for _ in range(JACK_NODES_NUMBER)]
cops_nodes = [NodeCops() for _ in range(COPS_NODES_NUMBER)]

with open("jack_edges.eg", "r") as f:
    raw = f.readlines()
    raw = list(map(lambda x: list(map(int, x.replace("\n", "").split(", "))), raw))

for new_edges in raw:
    base, *to = new_edges
    for edge in to:
        jack_add_edge(base, edge)

jack_edges = dict(zip(list(range(2, 16)), [[] for _ in range(2, 16)]))
for node in jack_nodes:
    jack_edges[len(node.edges)].append(node)

average = 0


with open("cops_edges.eg", "r") as f:
    raw = f.readlines()
    raw = list(map(lambda x: list(map(int, x.replace("\n", "").split(", "))), raw))

for new_edges in raw:
    base, *to = new_edges
    for edge in to:
        cops_add_edge(base, edge)


cops_edges = dict(zip(list(range(2, 8)), [[] for _ in range(2, 8)]))
for node in cops_nodes:
    cops_edges[len(node.edges)].append(node)


for i in sorted(node_edges.items(), key=lambda x: x[0]):
    print(f"{i[0]}: {len(i[1])}")
    average += len(i[1])*i[0]

average = round(average/JACK_NODES_NUMBER, 2)
print(f"Average edges per node: {average}")


# with open("jack.map", "wb") as f:
#     pickle.dump(jack_nodes, f)
#
# with open("cops.map", "wb") as f:
#     pickle.dump(cops_nodes, f)
