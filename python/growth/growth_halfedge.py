
import numpy as np

import pygame

MAX_VERTICES = 1000
EPSILON = 0.000001

class Mesh:
    def __init__(self):
        self.spring_len = 40
        self.elastic_constant = 0.05
        self.repulsion_distance = 200
        self.repulsion_strength = 2.0
        self.angle_maximization_force_strength = 1.0
        self.bulge_strength = 10.0

        self.face_idx = np.full((MAX_VERTICES), -1)
        self.face_half_edge = np.full((MAX_VERTICES), -1)

        self.vertex_idx = np.full((MAX_VERTICES), -1)
        self.vertex_half_edge = np.full((MAX_VERTICES), -1)
        self.vertex_pos = np.full((MAX_VERTICES, 2), -1, dtype=np.float32)

        self.half_edge_idx = np.full((MAX_VERTICES), -1)
        self.half_edge_twin = np.full((MAX_VERTICES), -1)
        self.half_edge_dest = np.full((MAX_VERTICES), -1)
        self.half_edge_face = np.full((MAX_VERTICES), -1)
        self.half_edge_next = np.full((MAX_VERTICES), -1)
        self.half_edge_prev = np.full((MAX_VERTICES), -1)

    def make_first_triangle(self, width, height):
        face_idx = 0

        vertex_a = 0
        vertex_b = 1
        vertex_c = 2
        self.vertex_idx[:3] = np.arange(3)
        self.vertex_pos[:3] = 0.5 * np.array([width, height]) + 40 * np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])
        
        half_edge_ab = 0
        half_edge_bc = 1
        half_edge_ca = 2
        half_edge_ba = 3
        half_edge_ac = 4
        half_edge_cb = 5

        self.vertex_half_edge[vertex_a] = half_edge_ab
        self.vertex_half_edge[vertex_b] = half_edge_bc
        self.vertex_half_edge[vertex_c] = half_edge_ca

        self.face_idx[face_idx] = face_idx
        self.face_half_edge[face_idx] = half_edge_ab

        self.half_edge_idx[:6] = np.arange(6)

        self.half_edge_twin[half_edge_ab] = half_edge_ba
        self.half_edge_twin[half_edge_ba] = half_edge_ab
        self.half_edge_twin[half_edge_bc] = half_edge_cb
        self.half_edge_twin[half_edge_cb] = half_edge_bc
        self.half_edge_twin[half_edge_ca] = half_edge_ac
        self.half_edge_twin[half_edge_ac] = half_edge_ca

        self.half_edge_face[half_edge_ab] = face_idx
        self.half_edge_face[half_edge_bc] = face_idx
        self.half_edge_face[half_edge_ca] = face_idx

        self.half_edge_dest[half_edge_ab] = vertex_b
        self.half_edge_dest[half_edge_bc] = vertex_c
        self.half_edge_dest[half_edge_ca] = vertex_a
        self.half_edge_dest[half_edge_ba] = vertex_a
        self.half_edge_dest[half_edge_ac] = vertex_c
        self.half_edge_dest[half_edge_cb] = vertex_b

        self.half_edge_next[half_edge_ab] = half_edge_bc
        self.half_edge_next[half_edge_bc] = half_edge_ca
        self.half_edge_next[half_edge_ca] = half_edge_ab
        self.half_edge_next[half_edge_ba] = half_edge_ac
        self.half_edge_next[half_edge_ac] = half_edge_cb
        self.half_edge_next[half_edge_cb] = half_edge_ba

        self.half_edge_prev[half_edge_ab] = half_edge_ca
        self.half_edge_prev[half_edge_bc] = half_edge_ab
        self.half_edge_prev[half_edge_ca] = half_edge_bc
        self.half_edge_prev[half_edge_ba] = half_edge_cb
        self.half_edge_prev[half_edge_ac] = half_edge_ba
        self.half_edge_prev[half_edge_cb] = half_edge_ac
        pass

    def add_external_triangle(self, vertex_a, vertex_b):
        new_face_idx = np.max(self.face_idx) + 1

        if new_face_idx >= MAX_VERTICES or np.max(self.half_edge_idx) + 3 >= MAX_VERTICES:
            print("Maximum number of vertices or half-edges reached.")
            return

        # get half edge already on the start of this face
        if self.half_edge_dest[self.vertex_half_edge[vertex_a]] == vertex_b:
            half_edge_ij = self.vertex_half_edge[vertex_a]
            vertex_i = vertex_a
            vertex_j = vertex_b
        else:
            half_edge_ij = self.vertex_half_edge[vertex_b]
            vertex_i = vertex_b
            vertex_j = vertex_a

        assert self.half_edge_face[half_edge_ij] != -1

        half_edge_ji = self.half_edge_twin[half_edge_ij]
        assert self.half_edge_face[half_edge_ji] == -1

        vertex_k = np.max(self.vertex_idx) + 1
        self.vertex_idx[vertex_k] = vertex_k
        self.vertex_pos[vertex_k] = np.mean([self.vertex_pos[vertex_i], self.vertex_pos[vertex_j]], axis=0)

        # create new face
        self.face_idx[new_face_idx] = new_face_idx
        self.face_half_edge[new_face_idx] = half_edge_ji

        # add the new vertex and 4 new half edges
        
        half_edge_ik = np.max(self.half_edge_idx) + 1
        half_edge_ki = half_edge_ik + 1

        half_edge_kj = half_edge_ik + 2
        half_edge_jk = half_edge_ki + 2

        half_edge_i_next = self.half_edge_next[half_edge_ji]
        half_edge_j_prev = self.half_edge_prev[half_edge_ji]

        # update existing half edge with new face
        self.half_edge_face[half_edge_ji] = new_face_idx
        self.half_edge_next[half_edge_ji] = half_edge_ik
        self.half_edge_prev[half_edge_ji] = half_edge_kj

        # update next and prev of existing half edge that's getting a new face
        self.half_edge_prev[half_edge_i_next] = half_edge_ki
        self.half_edge_next[half_edge_j_prev] = half_edge_jk

        # add first inner half edge
        self.half_edge_idx[half_edge_ik] = half_edge_ik
        self.half_edge_face[half_edge_ik] = new_face_idx
        self.half_edge_dest[half_edge_ik] = vertex_k
        self.half_edge_twin[half_edge_ik] = half_edge_ki
        self.half_edge_next[half_edge_ik] = half_edge_kj
        self.half_edge_prev[half_edge_ik] = half_edge_ji

        # add second inner half edge
        self.vertex_half_edge[vertex_k] = half_edge_kj
        self.half_edge_next[half_edge_ik] = half_edge_kj

        self.half_edge_idx[half_edge_kj] = half_edge_kj
        self.half_edge_face[half_edge_kj] = new_face_idx
        self.half_edge_dest[half_edge_kj] = vertex_j
        self.half_edge_twin[half_edge_kj] = half_edge_jk
        self.half_edge_prev[half_edge_kj] = half_edge_ik
        self.half_edge_next[half_edge_kj] = half_edge_ji

        # add first outer half edge
        self.half_edge_idx[half_edge_ki] = half_edge_ki
        self.half_edge_face[half_edge_ki] = -1
        self.half_edge_dest[half_edge_ki] = vertex_i
        self.half_edge_twin[half_edge_ki] = half_edge_ik
        self.half_edge_prev[half_edge_ki] = half_edge_jk
        self.half_edge_next[half_edge_ki] = half_edge_i_next

        # add second outer half edge
        self.half_edge_idx[half_edge_jk] = half_edge_jk
        self.half_edge_face[half_edge_jk] = -1
        self.half_edge_dest[half_edge_jk] = vertex_k
        self.half_edge_twin[half_edge_jk] = half_edge_kj
        self.half_edge_prev[half_edge_jk] = half_edge_j_prev
        self.half_edge_next[half_edge_jk] = half_edge_ki
        
        self.test()

    def add_internal_triangles(self, vertex_a, vertex_b):
        # find half edge from a to b
        half_edge_ab = self.vertex_half_edge[vertex_a]
        while self.half_edge_dest[half_edge_ab] != vertex_b:
            half_edge_ab = self.half_edge_next[self.half_edge_twin[half_edge_ab]]
            if half_edge_ab == self.vertex_half_edge[vertex_a]:
                print("No half edge found from vertex_a to vertex_b")
                return

        half_edge_ba = self.half_edge_twin[half_edge_ab]
        if self.half_edge_face[half_edge_ab] == -1 or self.half_edge_face[half_edge_ba] == -1:
            print("One of the half edges is on the boundary, cannot add internal triangle.")
            return
        
        vertex_c = self.half_edge_dest[self.half_edge_next[half_edge_ab]]
        vertex_d = self.half_edge_dest[self.half_edge_next[half_edge_ba]]
        assert vertex_c != vertex_d, "Cannot add internal triangles when both faces are the same."

        face_abc = self.half_edge_face[half_edge_ab]
        face_bad = self.half_edge_face[half_edge_ba]
        face_aec = face_abc
        face_bed = face_bad

        half_edge_ae = half_edge_ab
        half_edge_be = half_edge_ba

        face_ebc = np.max(self.face_idx) + 1
        face_ead = face_ebc + 1

        vertex_e = np.max(self.vertex_idx) + 1
        self.vertex_idx[vertex_e] = vertex_e
        self.vertex_pos[vertex_e] = np.mean([self.vertex_pos[vertex_a], self.vertex_pos[vertex_b]], axis=0)

        # create new face
        self.face_idx[face_aec] = face_aec
        self.face_idx[face_ead] = face_ead

        # edit face aec
        half_edge_ec = np.max(self.half_edge_idx) + 1
        half_edge_ea = half_edge_ec + 1
        half_edge_ce = half_edge_ea + 1
        half_edge_bc = self.half_edge_next[half_edge_ab]
        half_edge_ca = self.half_edge_next[half_edge_bc]
        half_edge_ad = self.half_edge_next[half_edge_ba]
        half_edge_db = self.half_edge_next[half_edge_ad]

        self.half_edge_dest[half_edge_ae] = vertex_e
        self.half_edge_next[half_edge_ae] = half_edge_ec
        self.half_edge_twin[half_edge_ae] = half_edge_ea

        self.half_edge_idx[half_edge_ec] = half_edge_ec
        self.half_edge_twin[half_edge_ec] = half_edge_ce
        self.half_edge_face[half_edge_ec] = face_aec
        self.half_edge_dest[half_edge_ec] = vertex_c
        self.half_edge_next[half_edge_ec] = half_edge_ca
        self.half_edge_prev[half_edge_ec] = half_edge_ae

        self.half_edge_prev[half_edge_ca] = half_edge_ec

        # edit face ebc
        half_edge_eb = half_edge_ce + 1
        self.vertex_half_edge[vertex_e] = half_edge_eb

        self.half_edge_idx[half_edge_eb] = half_edge_eb
        self.half_edge_twin[half_edge_eb] = half_edge_be
        self.half_edge_face[half_edge_eb] = face_ebc
        self.half_edge_dest[half_edge_eb] = vertex_b
        self.half_edge_next[half_edge_eb] = half_edge_bc
        self.half_edge_prev[half_edge_eb] = half_edge_ce

        self.half_edge_face[half_edge_bc] = face_ebc
        self.half_edge_next[half_edge_bc] = half_edge_ce
        self.half_edge_prev[half_edge_bc] = half_edge_eb

        self.half_edge_idx[half_edge_ce] = half_edge_ce
        self.half_edge_twin[half_edge_ce] = half_edge_ec
        self.half_edge_face[half_edge_ce] = face_ebc
        self.half_edge_dest[half_edge_ce] = vertex_e
        self.half_edge_next[half_edge_ce] = half_edge_eb
        self.half_edge_prev[half_edge_ce] = half_edge_bc

        # edit face ead
        half_edge_de = half_edge_eb + 1
        half_edge_ed = half_edge_de + 1

        self.half_edge_idx[half_edge_ea] = half_edge_ea
        self.half_edge_twin[half_edge_ea] = half_edge_ae
        self.half_edge_face[half_edge_ea] = face_ead
        self.half_edge_dest[half_edge_ea] = vertex_a
        self.half_edge_next[half_edge_ea] = half_edge_ad
        self.half_edge_prev[half_edge_ea] = half_edge_de

        self.half_edge_face[half_edge_ad] = face_ead
        self.half_edge_next[half_edge_ad] = half_edge_de
        self.half_edge_prev[half_edge_ad] = half_edge_ea

        self.half_edge_idx[half_edge_de] = half_edge_de
        self.half_edge_twin[half_edge_de] = half_edge_ed
        self.half_edge_face[half_edge_de] = face_ead
        self.half_edge_dest[half_edge_de] = vertex_e
        self.half_edge_next[half_edge_de] = half_edge_ea
        self.half_edge_prev[half_edge_de] = half_edge_ad

        # edit existing face bed
        self.half_edge_twin[half_edge_be] = half_edge_eb
        self.half_edge_face[half_edge_be] = face_bed
        self.half_edge_dest[half_edge_be] = vertex_e
        self.half_edge_next[half_edge_be] = half_edge_ed
        self.half_edge_prev[half_edge_be] = half_edge_db

        self.half_edge_idx[half_edge_ed] = half_edge_ed
        self.half_edge_twin[half_edge_ed] = half_edge_de
        self.half_edge_face[half_edge_ed] = face_bed
        self.half_edge_dest[half_edge_ed] = vertex_d
        self.half_edge_next[half_edge_ed] = half_edge_db
        self.half_edge_prev[half_edge_ed] = half_edge_be

        self.half_edge_next[half_edge_db] = half_edge_be
        self.half_edge_prev[half_edge_db] = half_edge_ed
        
        self.test()

    def make_hexagon(self, width, height):
        # add first two vertices
        self.make_first_triangle(width, height)
        
        for i in range(1):
            self.add_external_triangle(0, i+2)

    def test(self):
        assert np.all(self.half_edge_dest[self.vertex_half_edge] == self.half_edge_dest[self.half_edge_next[self.half_edge_prev[self.vertex_half_edge]]])
        assert np.all(self.half_edge_face[self.half_edge_idx] == self.half_edge_face[self.half_edge_next[self.half_edge_idx]])

        # assert np.all(np.bitwise_xor(self.half_edge_dest[self.half_edge_next[self.vertex_half_edge]] == -1, self.half_edge_dest[self.half_edge_next[self.vertex_half_edge]] != self.half_edge_dest[self.half_edge_next[self.half_edge_twin[self.vertex_half_edge]]]))

        max_iter = 100
        complete = np.zeros((self.vertex_idx.shape[0],), dtype=bool)
        iter = 0
        start_vertex = self.half_edge_dest[self.vertex_half_edge]
        this_vertex = start_vertex.copy()
        this_half_edge = self.vertex_half_edge.copy()
        while not np.all(complete) and iter < max_iter:
            next_half_edge = self.half_edge_next[self.half_edge_twin[this_half_edge]]
            next_vertex = self.half_edge_dest[next_half_edge]
            complete = complete | (next_vertex == start_vertex)
            this_half_edge = next_half_edge
            iter += 1
        assert iter < max_iter, "Looping over half edges did not complete in max iterations"

    def get_edge_count(self):
        complete = np.zeros((self.vertex_idx.shape[0],), dtype=bool)
        edge_count = np.zeros((self.vertex_idx.shape[0],), dtype=int)
        start_vertex = self.half_edge_dest[self.vertex_half_edge]
        this_half_edge = self.vertex_half_edge.copy()
        while not np.all(complete):
            next_half_edge = self.half_edge_next[self.half_edge_twin[this_half_edge]]
            next_vertex = self.half_edge_dest[next_half_edge]
            complete = complete | (next_vertex == start_vertex)
            this_half_edge = next_half_edge
            edge_count += ~complete.astype(int)
        return edge_count

    def update(self):
        if np.random.uniform() < 0.05:
            # randomly choose a vertex
            active_vertices = np.where(self.vertex_idx != -1)[0]

            # get edge count
            edge_count = self.get_edge_count()

            chosen_vertex = np.random.choice(active_vertices)
            # randomly choose one of its half edges
            chosen_half_edge = self.vertex_half_edge[chosen_vertex]
            # get the destination vertex of the half edge
            dest_vertex = self.half_edge_dest[chosen_half_edge]
            # check if this half edge has a twin (i.e. is on the boundary)
            if self.half_edge_face[chosen_half_edge] == -1 or self.half_edge_face[self.half_edge_twin[chosen_half_edge]] == -1:
                # add a triangle on this edge
                self.add_external_triangle(chosen_vertex, dest_vertex)
            else:
                self.add_internal_triangles(chosen_vertex, dest_vertex)

        spring_force = self.calculate_spring_force()
        repulsion_force = self.calculate_repulsion_force()
        angle_maximization_force = self.angle_maximization_force_strength * self.calculate_angle_maximization_force()
        bulge_force = self.bulge_strength * self.calculate_bulge_force()
        force = spring_force + repulsion_force + angle_maximization_force + bulge_force
        # Update vertex positions based on forces
        dt = 0.1
        self.vertex_pos += dt * force

    def get_edge_positions(self):
        edges = np.stack([self.half_edge_dest, self.half_edge_dest[self.half_edge_prev]])
        edge_pos = self.vertex_pos[edges]
        return edges, edge_pos

    def calculate_spring_force(self):
        edges, edge_pos = self.get_edge_positions()
        edge_vector = edge_pos[0, :] - edge_pos[1, :]
        edge_lengths = np.linalg.norm(edge_vector, axis=1)
        edge_force = -1 * (edge_lengths[:, np.newaxis] - self.spring_len) * self.elastic_constant * (edge_vector / (EPSILON + edge_lengths[:, np.newaxis]))
        vertex_force = np.zeros_like(self.vertex_pos)
        np.add.at(vertex_force, edges[0], edge_force)
        np.add.at(vertex_force, edges[1], -edge_force)
        return vertex_force
    
    def calculate_repulsion_force(self):
        vertex_diff = self.vertex_pos[:, np.newaxis, :] - self.vertex_pos[np.newaxis, :, :]
        vertex_dist = np.sqrt(np.sum(np.square(vertex_diff), axis=-1))
        repulsion_force = self.repulsion_strength * np.square(np.maximum(0, (self.repulsion_distance - vertex_dist) / self.repulsion_distance))[..., np.newaxis] * (vertex_diff / (EPSILON + vertex_dist[:, :, np.newaxis]))
        repulsion_force = repulsion_force.sum(axis=1)
        return repulsion_force
    
    def calculate_angle_maximization_force(self):
        forces = np.zeros_like(self.vertex_pos)
        
        start_half_edge = self.vertex_half_edge.copy()
        start_vertex = self.half_edge_dest[start_half_edge]
        vertex_a = start_vertex.copy()
        half_edge_a = start_half_edge.copy()
        iterating = np.ones_like(half_edge_a)
        while True:
            half_edge_b = self.half_edge_next[self.half_edge_twin[half_edge_a]]
            vertex_b = self.half_edge_dest[half_edge_b]
            iterating = iterating & (vertex_b != start_vertex)
            if not np.any(iterating):
                break
            vertex_a_pos = self.vertex_pos[vertex_a]
            vertex_b_pos = self.vertex_pos[vertex_b]
            vertex_a_dir = vertex_a_pos - self.vertex_pos
            vertex_b_dir = vertex_b_pos - self.vertex_pos
            angle = np.arccos(np.clip(np.sum(vertex_a_dir * vertex_b_dir, axis=-1) / (np.linalg.norm(vertex_a_dir, axis=-1) * np.linalg.norm(vertex_b_dir, axis=-1) + EPSILON), -1.0, 1.0))
            force_mag = np.power(np.maximum(np.pi/2 - angle / (np.pi/2), 0), 3)
            a_z_dir = np.array([vertex_a_dir[:, 0], vertex_a_dir[:, 1], np.zeros_like(vertex_a_dir[:, 0])]).T
            b_z_dir = np.array([vertex_b_dir[:, 0], vertex_b_dir[:, 1], np.zeros_like(vertex_b_dir[:, 0])]).T
            cross_prod = np.cross(a_z_dir, b_z_dir)
            force_a_dir = np.cross(a_z_dir, cross_prod)[:, :2]
            force_b_dir = np.cross(cross_prod, b_z_dir)[:, :2]
            force_a_dir /= np.linalg.norm(force_a_dir, axis=-1)[:, np.newaxis] + EPSILON
            force_b_dir /= np.linalg.norm(force_b_dir, axis=-1)[:, np.newaxis] + EPSILON

            forces[vertex_a] += (force_mag[:, np.newaxis] * force_a_dir) * iterating[:, np.newaxis]
            forces[vertex_b] += (force_mag[:, np.newaxis] * force_b_dir) * iterating[:, np.newaxis]
            half_edge_a = half_edge_b
            vertex_a = vertex_b

        return forces
    
    def calculate_bulge_force(self):
        # draw edge normals
        border_half_edges = np.where((self.half_edge_face[self.half_edge_idx] == -1) & (self.half_edge_idx != -1))[0]
        edges = np.stack([self.half_edge_dest[border_half_edges], self.half_edge_dest[self.half_edge_twin[border_half_edges]]])
        edge_pos = self.vertex_pos[edges]
        z_orth = np.array([[0, 0, 1]] * edge_pos.shape[1])
        edge_vector = edge_pos[1, :] - edge_pos[0, :]
        edge_vector_3d = np.array([edge_vector[:, 0], edge_vector[:, 1], np.zeros_like(edge_vector[:, 0])]).T
        edge_normal_3d = np.cross(edge_vector_3d, z_orth)
        edge_normal_2d = edge_normal_3d[:, :2]
        edge_normal_2d /= np.linalg.norm(edge_normal_2d, axis=-1)[:, np.newaxis] + EPSILON
        
        vertex_force = np.zeros_like(self.vertex_pos)
        np.add.at(vertex_force, edges[0], edge_normal_2d)
        np.add.at(vertex_force, edges[1], edge_normal_2d)
        vertex_force /= np.linalg.norm(vertex_force, axis=-1)[:, np.newaxis] + EPSILON
        return vertex_force


    def draw(self, screen):
        screen.fill((0, 0, 0))
        
        # Draw connections
        _, edge_pos = self.get_edge_positions()
        for i in range(edge_pos.shape[1]):
            link_start = tuple(map(float, edge_pos[0, i]))
            link_end = tuple(map(float, edge_pos[1, i]))
            pygame.draw.line(screen, (50, 50, 50), link_start, link_end, 2)
        
        # Draw cells
        active_cells = np.where(self.vertex_idx != -1)[0]
        active_cell_positions = self.vertex_pos[active_cells]
        for cell_position in active_cell_positions:
            pos = tuple(map(float, cell_position))
            pygame.draw.circle(screen, (255, 255, 255), pos, 5)
        
        pygame.display.flip()


def main():
    np.random.seed(42)

    width, height = 1400, 1400

    mesh = Mesh()

    mesh.make_first_triangle(width, height)
    mesh.test()
    mesh.add_external_triangle(0, 2)

    clock = pygame.time.Clock()
    pygame.init()
    screen = pygame.display.set_mode((width, height))

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        mesh.update()
        mesh.draw(screen)
        clock.tick(60)

    pygame.quit()

if __name__ == '__main__':
    main()