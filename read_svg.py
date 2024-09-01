from xml.dom import minidom
from tqdm import tqdm
import numpy as np
import pygame


class Node:
    id = 196

    def __init__(self, node_type, coordinates, node_id=None):
        if node_id is None:
            self.id = self.__class__.id  # Jack spots have id corresponding to their numbers
            self.__class__.id += 1
        else:
            self.id = node_id
        self.type = node_type
        self.coordinates = coordinates
        self.connected_paths = []
        self.connected_cops = []
        self.connected_jack = []

    def is_connected(self, other):
        for pos1 in self.coordinates:
            for pos2 in other.coordinates:
                if self.distance(pos1, pos2) < 5:
                    return True
        return False

    @staticmethod
    def distance(pos1, pos2):
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        return (dx**2 + dy**2)**0.5

    def new_connection(self, other):
        if self is not other and self not in other.connected_paths:
            self.connected_paths.append(other)
            other.connected_paths.append(self)

    @classmethod
    def from_path(cls, path):
        path_string = path.getAttribute('d')
        args = path_string.split(" ")
        coordinates = None
        current_mode = ""
        points_n = 1
        for arg in args:
            if len(arg) == 1:  # All the modes are single letters
                current_mode = arg
                continue
            else:  # If it's not a mode it must be a numeric value (single or a pair)
                if arg.find(",") == -1:  # Pairs of values are split by a ","
                    values = float(arg)
                else:
                    values = list(map(float, arg.split(",")))
                if coordinates is None:  # First value always is absolute
                    coordinates = [values.copy()]
                    continue

            # This explains it better than I could: https://developer.mozilla.org/en-US/docs/Web/SVG/Tutorial/Paths
            if current_mode in ["M", "L"]:
                coordinates.append(values.copy())
            elif current_mode in ["m", "l"]:
                coordinates.append(coordinates[points_n - 1].copy())
                coordinates[points_n][0] = round(coordinates[points_n][0] + values[0], 8)
                coordinates[points_n][1] = round(coordinates[points_n][1] + values[1], 8)
            elif current_mode == "H":
                coordinates.append(coordinates[points_n - 1].copy())
                coordinates[points_n][0] = values
            elif current_mode == "h":
                coordinates.append(coordinates[points_n - 1].copy())
                coordinates[points_n][0] = round(coordinates[points_n][0] + values, 8)
            elif current_mode == "V":
                coordinates.append(coordinates[points_n - 1].copy())
                coordinates[points_n][1] = values
            elif current_mode == "v":
                coordinates.append(coordinates[points_n - 1].copy())
                coordinates[points_n][1] = round(coordinates[points_n][1] + values, 8)
            else:
                raise RuntimeError("Unexpected path mode")
            points_n += 1
        tmp = [list(map(lambda x: round(x, 2), coordinates_set)) for coordinates_set in coordinates]
        return cls("path", tmp)

    def purify_paths(self):
        for node in self.connected_paths.copy():
            if node.type != "path":
                if node.type in ["jack", "jack_kill"]:
                    self.connected_jack.append(node)
                if node.type in ["cops", "cops_spawn"]:
                    self.connected_cops.append(node)
                self.connected_paths.remove(node)

    @classmethod
    def from_cops_spots(cls, rect):
        x = float(rect.getAttribute('x')) + float(rect.getAttribute('width')) / 2
        y = float(rect.getAttribute('y')) + float(rect.getAttribute('height')) / 2
        node_type = "cops_spawn" if rect.getAttribute("style").find("stroke:#ffff00") != -1 else "cops"
        return cls(node_type, [[round(x, 2), round(y, 2)]])

    @classmethod
    def from_jack_spot(cls, jack_spot):
        scale = 0.26458333  # "matrix(0.26458333,0,0,0.26458333,9.26376,-28.409268)"
        dx = 9.26376        # Hard coded from values in SVG file
        dy = -28.409268
        ellipse = jack_spot.getElementsByTagName('ellipse')[0]
        text = jack_spot.getElementsByTagName('tspan')[0]
        node_type = "jack_kill" if ellipse.getAttribute("style").find("fill:#ff0000") != -1 else "jack"
        x = round(float(ellipse.getAttribute('cx')) * scale + dx, 2)
        y = round(float(ellipse.getAttribute('cy')) * scale + dy, 2)
        jack_number = int(text.firstChild.nodeValue)
        return cls(node_type, [[x, y]], node_id=jack_number)

    def draw(self, screen, set_color=None):
        if self.type in ["jack", "jack_kill"]:
            color = (0, 0, 0) if self.type == "jack" else (255, 0, 0)
            color = color if set_color is None else set_color
            pygame.draw.circle(screen, color, self.coordinates[0], 4)
        if self.type in ["cops", "cops_spawn"]:
            color = (0, 0, 0) if self.type == "cops" else (255, 255, 0)
            color = color if set_color is None else set_color
            pygame.draw.rect(screen, color, pygame.Rect(self.coordinates[0][0]-1, self.coordinates[0][1]-1, 4, 4))
        if self.type == "path":
            color = (0, 0, 0)
            color = color if set_color is None else set_color
            pygame.draw.lines(screen, color, False, self.coordinates)

    def find_cops(self, searched):
        for node in searched.connected_paths:
            if node is not self and (searched is self or node not in self.connected_paths):
                if node.type in ["cops", "cops_spawn"]:
                    self.connected_cops.append(node)
                    return
                elif node.type in ["jack", "jack_kill"]:
                    if self.type in ["cops", "cops_spawn"]:
                        continue
                    return
                else:
                    if node not in self.connected_paths:
                        self.connected_paths.append(node)
                    self.find_cops(node)
        if self in self.connected_cops:
            self.connected_cops.remove(self)

    def find_jack(self, searched):
        for node in searched.connected_paths:
            if node is not self and (searched is self or node not in self.connected_paths):
                if node.type in ["jack", "jack_kill"]:
                    self.connected_jack.append(node)
                    return
                elif node.type in ["cops", "cops_spawn"]:
                    if self.type in ["jack", "jack_kill"]:
                        continue
                    return
                else:
                    if node not in self.connected_paths:
                        self.connected_paths.append(node)
                    self.find_jack(node)
        if self in self.connected_jack:
            self.connected_jack.remove(self)

    def __str__(self):
        return "Id: " + str(self.id)


svg_file = "Mapa_v5.svg"
doc = minidom.parse(svg_file)  # parseString also exists


paths = doc.getElementsByTagName('path')
cops = doc.getElementsByTagName('rect')
jack = [spot for spot in doc.getElementsByTagName("g") if spot.getAttribute("id").find("layer") == -1]

jack_nodes = sorted([Node.from_jack_spot(spot) for spot in jack], key=lambda x: x.id)
cops_nodes = [Node.from_cops_spots(spot) for spot in cops]
path_nodes = [Node.from_path(path_string) for path_string in paths]

all_nodes = [*jack_nodes, *cops_nodes, *path_nodes]

for node in all_nodes:
    for path in path_nodes:
        if node.is_connected(path):
            node.new_connection(path)

for node in tqdm([*jack_nodes, *cops_nodes]):  # [*jack_nodes, *cops_nodes]:
    node.find_cops(node)
    node.find_jack(node)

for node in tqdm(path_nodes):
    node.purify_paths()


print(len(jack_nodes))
print(len(cops_nodes))

jack_matrix = np.zeros((195, 195))
cops_matrix = np.zeros((234, 234))
neighbour_matrix = np.zeros((195, 234))

for node in jack_nodes:
    for connected in node.connected_jack:
        jack_matrix[node.id - 1, connected.id - 1] = 1
        jack_matrix[connected.id - 1, node.id - 1] = 1

for node in cops_nodes:
    for connected in node.connected_cops:
        n1, n2 = node.id - 196, connected.id - 196
        cops_matrix[node.id - 196, connected.id - 196] = 1
        cops_matrix[connected.id - 196, node.id - 196] = 1

for node in jack_nodes:
    for connected in node.connected_cops:
        neighbour_matrix[node.id - 1, connected.id - 196] = 1

a=0