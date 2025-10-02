
import glm
import moderngl
import numpy as np
import pygame

MAX_VERTICES = 1000
EPSILON = 0.000001

class Mesh:
    def __init__(self, num_dims=3):
        self.spring_len = 40
        self.elastic_constant = 0.1
        self.repulsion_distance = 200
        self.repulsion_strength = 2.0
        self.bulge_strength = 10.0
        self.planar_strength = 0.1
        self.wireframe_mode = False

        self.face_idx = np.full((MAX_VERTICES), -1)
        self.face_half_edge = np.full((MAX_VERTICES), -1)

        self.vertex_idx = np.full((MAX_VERTICES), -1)
        self.vertex_half_edge = np.full((MAX_VERTICES), -1)
        self.vertex_pos = np.full((MAX_VERTICES, num_dims), -1, dtype=np.float32)

        self.half_edge_idx = np.full((MAX_VERTICES), -1)
        self.half_edge_twin = np.full((MAX_VERTICES), -1)
        self.half_edge_dest = np.full((MAX_VERTICES), -1)
        self.half_edge_face = np.full((MAX_VERTICES), -1)
        self.half_edge_next = np.full((MAX_VERTICES), -1)
        self.half_edge_prev = np.full((MAX_VERTICES), -1)

    def init_shader(self):
        self.ctx = moderngl.get_context()

        self.program = self.ctx.program(
            vertex_shader='''
                #version 330 core

                uniform mat4 camera;

                layout (location = 0) in vec3 in_vertex;
                layout (location = 1) in vec3 in_normal;

                out vec3 world_pos;
                out vec3 normal;

                void main() {
                    world_pos = in_vertex;
                    normal = normalize(in_normal);
                    gl_Position = camera * vec4(in_vertex, 1.0);
                }
            ''',
            fragment_shader='''
                #version 330 core

                uniform vec3 light_pos;
                uniform vec3 light_color;
                uniform vec3 ambient_color;

                in vec3 world_pos;
                in vec3 normal;

                layout (location = 0) out vec4 out_color;

                void main() {
                    vec3 light_dir = normalize(light_pos - world_pos);
                    vec3 face_normal = normalize(normal);

                    // For two-sided lighting, use absolute value of dot product
                    float diff = abs(dot(face_normal, light_dir));

                    vec3 ambient = ambient_color;
                    vec3 diffuse = diff * light_color;

                    vec3 result = ambient + diffuse;
                    out_color = vec4(result, 1.0);
                }
            ''',
        )

        self.vbo = self.ctx.buffer(reserve=MAX_VERTICES * 3 * 4)
        self.nbo = self.ctx.buffer(reserve=MAX_VERTICES * 3 * 4)  # normal buffer
        self.ibo = self.ctx.buffer(reserve=MAX_VERTICES * 3 * 4)  # index buffer
        self.vao = self.ctx.vertex_array(self.program, [(self.vbo, '3f', 'in_vertex'), (self.nbo, '3f', 'in_normal')], index_buffer=self.ibo)

    def make_first_triangle(self, width, height):
        face_idx = 0

        vertex_a = 0
        vertex_b = 1
        vertex_c = 2
        self.vertex_idx[:3] = np.arange(3)
        self.vertex_pos[:3] = 0.5 * np.array([width, height, 0]) + 40 * np.array([[0, 0, 0], [1, 0, 0], [0.5, np.sqrt(3)/2, 0]])
        
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

        # get half edge from a to b
        half_edge_ab = self.vertex_half_edge[vertex_a]
        while self.half_edge_dest[half_edge_ab] != vertex_b:
            half_edge_ab = self.half_edge_next[self.half_edge_twin[half_edge_ab]]
            if half_edge_ab == self.vertex_half_edge[vertex_a]:
                print("No half edge found from vertex_a to vertex_b")
                return

        assert self.half_edge_face[half_edge_ab] == -1

        vertex_c = np.max(self.vertex_idx) + 1
        self.vertex_idx[vertex_c] = vertex_c
        self.vertex_pos[vertex_c] = np.mean([self.vertex_pos[vertex_a], self.vertex_pos[vertex_b]], axis=0)

        # create new face
        self.face_idx[new_face_idx] = new_face_idx
        self.face_half_edge[new_face_idx] = half_edge_ab

        # add the new vertex and 4 new half edges
        
        half_edge_bc = np.max(self.half_edge_idx) + 1
        half_edge_ca = half_edge_bc + 1

        half_edge_cb = half_edge_ca + 1
        half_edge_ac = half_edge_cb + 1

        half_edge_b_next = self.half_edge_next[half_edge_ab]
        half_edge_a_prev = self.half_edge_prev[half_edge_ab]

        # update existing half edge with new face
        self.half_edge_face[half_edge_ab] = new_face_idx
        self.half_edge_next[half_edge_ab] = half_edge_bc
        self.half_edge_prev[half_edge_ab] = half_edge_ca

        # update next and prev of existing half edge that's getting a new face
        self.half_edge_prev[half_edge_b_next] = half_edge_cb
        self.half_edge_next[half_edge_a_prev] = half_edge_ac

        # add first inner half edge
        self.half_edge_idx[half_edge_bc] = half_edge_bc
        self.half_edge_face[half_edge_bc] = new_face_idx
        self.half_edge_dest[half_edge_bc] = vertex_c
        self.half_edge_twin[half_edge_bc] = half_edge_cb
        self.half_edge_next[half_edge_bc] = half_edge_ca
        self.half_edge_prev[half_edge_bc] = half_edge_ab

        # add second inner half edge
        self.vertex_half_edge[vertex_c] = half_edge_ca

        self.half_edge_idx[half_edge_ca] = half_edge_ca
        self.half_edge_face[half_edge_ca] = new_face_idx
        self.half_edge_dest[half_edge_ca] = vertex_a
        self.half_edge_twin[half_edge_ca] = half_edge_ac
        self.half_edge_prev[half_edge_ca] = half_edge_bc
        self.half_edge_next[half_edge_ca] = half_edge_ab

        # add first outer half edge
        self.half_edge_idx[half_edge_cb] = half_edge_cb
        self.half_edge_face[half_edge_cb] = -1
        self.half_edge_dest[half_edge_cb] = vertex_b
        self.half_edge_twin[half_edge_cb] = half_edge_bc
        self.half_edge_prev[half_edge_cb] = half_edge_ac
        self.half_edge_next[half_edge_cb] = half_edge_b_next

        # add second outer half edge
        self.half_edge_idx[half_edge_ac] = half_edge_ac
        self.half_edge_face[half_edge_ac] = -1
        self.half_edge_dest[half_edge_ac] = vertex_c
        self.half_edge_twin[half_edge_ac] = half_edge_ca
        self.half_edge_prev[half_edge_ac] = half_edge_a_prev
        self.half_edge_next[half_edge_ac] = half_edge_cb

        self.test()
        
    def add_internal_edge_triangle(self, vertex_a, vertex_b):
        # assert that there is a half edge from a to b and its twin is on the boundary
        half_edge_ab = self.vertex_half_edge[vertex_a]
        while self.half_edge_dest[half_edge_ab] != vertex_b:
            half_edge_ab = self.half_edge_next[self.half_edge_twin[half_edge_ab]]
            if half_edge_ab == self.vertex_half_edge[vertex_a]:
                print("No half edge found from vertex_a to vertex_b")
                return

        half_edge_ba = self.half_edge_twin[half_edge_ab]
        assert self.half_edge_face[half_edge_ab] != -1, "Half edge ab is not internal"
        assert self.half_edge_face[half_edge_ba] == -1, "Half edge ba is not on boundary"

        half_edge_bc = self.half_edge_next[half_edge_ab]
        half_edge_ca = self.half_edge_next[half_edge_bc]

        vertex_c = self.half_edge_dest[half_edge_bc]
        face_abc = self.half_edge_face[half_edge_ab]

        vertex_d = np.max(self.vertex_idx) + 1
        face_dbc = np.max(self.face_idx) + 1
        half_edge_ad = half_edge_ab
        half_edge_db = np.max(self.half_edge_idx) + 1
        half_edge_cd = half_edge_db + 1
        half_edge_dc = half_edge_cd + 1
        half_edge_bd = half_edge_ba
        half_edge_da = half_edge_dc + 1
        face_adc = face_abc

        self.face_idx[face_dbc] = face_dbc
        self.face_half_edge[face_dbc] = half_edge_bc

        self.face_half_edge[face_adc] = half_edge_ca

        self.vertex_idx[vertex_d] = vertex_d
        self.vertex_pos[vertex_d] = np.mean([self.vertex_pos[vertex_a], self.vertex_pos[vertex_b]], axis=0)
        self.vertex_half_edge[vertex_d] = half_edge_db

        self.half_edge_face[half_edge_ad] = face_adc
        self.half_edge_dest[half_edge_ad] = vertex_d
        self.half_edge_twin[half_edge_ad] = half_edge_da
        self.half_edge_next[half_edge_ad] = half_edge_dc
        self.half_edge_prev[half_edge_ad] = half_edge_ca

        self.half_edge_idx[half_edge_db] = half_edge_db
        self.half_edge_face[half_edge_db] = face_dbc
        self.half_edge_dest[half_edge_db] = vertex_b
        self.half_edge_twin[half_edge_db] = half_edge_bd
        self.half_edge_next[half_edge_db] = half_edge_bc
        self.half_edge_prev[half_edge_db] = half_edge_cd

        half_edge_ba_next = self.half_edge_next[half_edge_ba]
        half_edge_ba_prev = self.half_edge_prev[half_edge_ba]
        self.half_edge_prev[half_edge_ba_next] = half_edge_da
        self.half_edge_next[half_edge_ba_prev] = half_edge_bd

        self.half_edge_dest[half_edge_bd] = vertex_d
        self.half_edge_twin[half_edge_bd] = half_edge_db
        self.half_edge_next[half_edge_bd] = half_edge_da

        self.half_edge_idx[half_edge_da] = half_edge_da
        self.half_edge_face[half_edge_da] = -1
        self.half_edge_dest[half_edge_da] = vertex_a
        self.half_edge_twin[half_edge_da] = half_edge_ad
        self.half_edge_next[half_edge_da] = half_edge_ba_next
        self.half_edge_prev[half_edge_da] = half_edge_bd

        self.half_edge_idx[half_edge_dc] = half_edge_dc
        self.half_edge_face[half_edge_dc] = face_adc
        self.half_edge_dest[half_edge_dc] = vertex_c
        self.half_edge_twin[half_edge_dc] = half_edge_cd
        self.half_edge_next[half_edge_dc] = half_edge_ca
        self.half_edge_prev[half_edge_dc] = half_edge_ad

        self.half_edge_idx[half_edge_cd] = half_edge_cd
        self.half_edge_face[half_edge_cd] = face_dbc
        self.half_edge_dest[half_edge_cd] = vertex_d
        self.half_edge_twin[half_edge_cd] = half_edge_dc
        self.half_edge_next[half_edge_cd] = half_edge_db
        self.half_edge_prev[half_edge_cd] = half_edge_bc

        self.half_edge_face[half_edge_ca] = face_adc
        self.half_edge_next[half_edge_ca] = half_edge_ad
        self.half_edge_prev[half_edge_ca] = half_edge_dc

        self.half_edge_face[half_edge_bc] = face_dbc
        self.half_edge_next[half_edge_bc] = half_edge_cd
        self.half_edge_prev[half_edge_bc] = half_edge_db
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
        self.face_idx[face_ebc] = face_ebc
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

        self.face_half_edge[face_ebc] = half_edge_eb
        self.face_half_edge[face_ead] = half_edge_ea
        
        self.test()

    def make_hexagon(self, width, height):
        # add first two vertices
        self.make_first_triangle(width, height)
        
        for i in range(1):
            self.add_external_triangle(0, i+2)

    def test(self):
        assert np.all(self.half_edge_dest[self.vertex_half_edge] == self.half_edge_dest[self.half_edge_next[self.half_edge_prev[self.vertex_half_edge]]])
        assert np.all(self.half_edge_face[self.half_edge_idx] == self.half_edge_face[self.half_edge_next[self.half_edge_idx]]), f"Half edge {np.where(self.half_edge_face[self.half_edge_idx] != self.half_edge_face[self.half_edge_next[self.half_edge_idx]])[0].tolist()} have different faces than their next half edges"
        assert np.all(self.half_edge_idx == self.half_edge_idx[self.half_edge_next[self.half_edge_prev[self.half_edge_idx]]]), f"Half edge {np.where(self.half_edge_idx != self.half_edge_idx[self.half_edge_next[self.half_edge_prev[self.half_edge_idx]]])[0].tolist()} do not loop back to themselves when going next then prev"

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
            edge_count += ~complete
        return edge_count

    def update(self):
        spring_force = self.calculate_spring_force()
        repulsion_force_mag, repulsion_force = self.calculate_repulsion_force()
        repulsion_force *= self.repulsion_strength
        bulge_force = self.bulge_strength * self.calculate_bulge_force()
        planar_force = self.planar_strength * self.calculate_planar_force()
        force = spring_force + repulsion_force + bulge_force + planar_force
        # Update vertex positions based on forces
        dt = 0.1
        self.vertex_pos += dt * force

        self.generate_new_triangles(repulsion_force_mag)
        self.refine_mesh()

    def refine_mesh(self):
        # iterate over edges
        edges = np.stack([self.half_edge_dest, self.half_edge_dest[self.half_edge_twin]], axis=-1)

        filter_edges = (edges[:, 0] < edges[:, 1]) & (self.half_edge_face != -1) & (self.half_edge_face[self.half_edge_twin] != -1)

        # get two other vertices
        vertex_c = self.half_edge_dest[self.half_edge_next]
        vertex_d = self.half_edge_dest[self.half_edge_next[self.half_edge_twin]]

        vertex_quads = np.stack([edges[:, 0], edges[:, 1], vertex_c, vertex_d], axis=-1)
        vertex_quads = vertex_quads[filter_edges]
        quad_half_edges = self.half_edge_idx[filter_edges]

        no_flip_valence_weights = np.full((4,), 6)
        flip_valence_weights = np.array([7, 7, 5, 5])

        for quad_half_edge, vertex_quad in zip(quad_half_edges, vertex_quads):
            # get vertex adjacency
            edge_count = self.get_edge_count()

            vertex_quad_edge_counts = edge_count[vertex_quad]
        
            no_flip_valence = np.sum(np.square(vertex_quad_edge_counts - no_flip_valence_weights), axis=-1)
            flip_valence = np.sum(np.square(vertex_quad_edge_counts - flip_valence_weights), axis=-1)

            if flip_valence < no_flip_valence:
                self.flip_edge(quad_half_edge)

    def flip_edge(self, half_edge):
        half_edge_ab = half_edge
        half_edge_ba = self.half_edge_twin[half_edge_ab]
        if self.half_edge_face[half_edge_ab] == -1 or self.half_edge_face[half_edge_ba] == -1:
            print("One of the half edges is on the boundary, cannot flip edge.")
            return
        
        vertex_a = self.half_edge_dest[half_edge_ba]
        vertex_b = self.half_edge_dest[half_edge_ab]

        half_edge_bc = self.half_edge_next[half_edge_ab]
        vertex_c = self.half_edge_dest[half_edge_bc]
        half_edge_ad = self.half_edge_next[half_edge_ba]
        vertex_d = self.half_edge_dest[half_edge_ad]

        half_edge_ca = self.half_edge_next[half_edge_bc]
        half_edge_db = self.half_edge_next[half_edge_ad]

        face_abc = self.half_edge_face[half_edge_ab]
        face_bad = self.half_edge_face[half_edge_ba]

        face_adc = face_abc
        face_bcd = face_bad

        half_edge_dc = half_edge_ab
        half_edge_cd = half_edge_ba

        self.face_half_edge[face_adc] = half_edge_ca
        self.face_half_edge[face_bcd] = half_edge_db

        self.vertex_half_edge[vertex_a] = half_edge_ad
        self.vertex_half_edge[vertex_b] = half_edge_bc

        # change surrounding half edges that aren't moving
        self.half_edge_next[half_edge_ca] = half_edge_ad
        self.half_edge_prev[half_edge_ca] = half_edge_dc
        self.half_edge_face[half_edge_ca] = face_adc

        self.half_edge_next[half_edge_db] = half_edge_bc
        self.half_edge_prev[half_edge_db] = half_edge_cd
        self.half_edge_face[half_edge_db] = face_bcd

        self.half_edge_next[half_edge_ad] = half_edge_dc
        self.half_edge_prev[half_edge_ad] = half_edge_ca
        self.half_edge_face[half_edge_ad] = face_adc

        self.half_edge_next[half_edge_bc] = half_edge_cd
        self.half_edge_prev[half_edge_bc] = half_edge_db
        self.half_edge_face[half_edge_bc] = face_bcd

        # change moving half edges
        self.half_edge_dest[half_edge_dc] = vertex_c
        self.half_edge_next[half_edge_dc] = half_edge_ca
        self.half_edge_prev[half_edge_dc] = half_edge_ad
        self.half_edge_face[half_edge_dc] = face_adc

        self.half_edge_dest[half_edge_cd] = vertex_d
        self.half_edge_next[half_edge_cd] = half_edge_db
        self.half_edge_prev[half_edge_cd] = half_edge_bc
        self.half_edge_face[half_edge_cd] = face_bcd

    def generate_new_triangles(self, repulsion_force_mag):
        repulsion_force_mag_norm = repulsion_force_mag / np.min(repulsion_force_mag[repulsion_force_mag > 0])
        self.vertex_suitability = 1 / (1 + repulsion_force_mag_norm ** 2)
        self.vertex_suitability *= (self.vertex_idx != -1).astype(float)

        if np.random.uniform() < 0.1:

            # randomly choose a vertex
            active_vertices = np.where(self.vertex_idx != -1)[0]
            active_vertex_suitability = self.vertex_suitability[active_vertices]

            chosen_vertex = np.random.choice(active_vertices)#, p=active_vertex_suitability / np.sum(active_vertex_suitability))
            # randomly choose one of its half edges
            chosen_half_edge = self.vertex_half_edge[chosen_vertex]
            # get the destination vertex of the half edge
            dest_vertex = self.half_edge_dest[chosen_half_edge]
            # check if this half edge has a twin (i.e. is on the boundary)
            chosen_on_boundary = self.half_edge_face[chosen_half_edge] == -1
            chosen_twin_on_boundary = self.half_edge_face[self.half_edge_twin[chosen_half_edge]] == -1
            if chosen_on_boundary or chosen_twin_on_boundary:
                # add a triangle on this edge
                # decide to add external triangle or internal edge triangle
                if np.random.uniform() < 0.5:
                    if chosen_on_boundary:
                        self.add_external_triangle(chosen_vertex, dest_vertex)
                    else:
                        self.add_external_triangle(dest_vertex, chosen_vertex)
                else:
                    if chosen_on_boundary:
                        self.add_internal_edge_triangle(dest_vertex, chosen_vertex)
                    else:
                        self.add_internal_edge_triangle(chosen_vertex, dest_vertex)
            else:
                self.add_internal_triangles(chosen_vertex, dest_vertex)

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
        repulsion_force = np.power(np.maximum(0, (self.repulsion_distance - vertex_dist) / self.repulsion_distance), 2)[..., np.newaxis] * (vertex_diff / (EPSILON + vertex_dist[:, :, np.newaxis]))
        repulsion_force *= np.outer(self.vertex_idx != -1, self.vertex_idx != -1)[:, :, np.newaxis]
        repulsion_force_mag = np.linalg.norm(repulsion_force, axis=-1)
        repulsion_force = repulsion_force.sum(axis=1)
        repulsion_force_mag = repulsion_force_mag.sum(axis=1)
        return repulsion_force_mag, repulsion_force
    
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
    
    def calculate_planar_force(self):
        # for each vertex, sum its neighbour positions and get its edge count
        vertex_sum = np.zeros_like(self.vertex_pos)
        edge_count = np.zeros((self.vertex_idx.shape[0],), dtype=int)
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
            vertex_sum[vertex_a] += self.vertex_pos[vertex_b] * iterating[:, np.newaxis]
            edge_count += iterating
            half_edge_a = half_edge_b
            vertex_a = vertex_b

        neighbour_avg = vertex_sum / (edge_count[:, np.newaxis] + EPSILON)
        planar_force = (neighbour_avg - self.vertex_pos) * (self.vertex_idx != -1)[:, np.newaxis]
        return planar_force
    
    def calculate_bulge_force(self):
        # draw edge normals
        border_half_edges = np.where((self.half_edge_face[self.half_edge_idx] == -1) & (self.half_edge_idx != -1))[0]
        edges = np.stack([self.half_edge_dest[border_half_edges], self.half_edge_dest[self.half_edge_twin[border_half_edges]]])
        edge_pos = self.vertex_pos[edges]
        z_orth = np.array([[0, 0, 1]] * edge_pos.shape[1])
        edge_vector = edge_pos[1, :] - edge_pos[0, :]
        edge_normal = np.cross(edge_vector, z_orth)
        edge_normal /= np.linalg.norm(edge_normal, axis=-1)[:, np.newaxis] + EPSILON
        
        vertex_force = np.zeros_like(self.vertex_pos)
        np.add.at(vertex_force, edges[0], edge_normal)
        np.add.at(vertex_force, edges[1], edge_normal)
        vertex_force /= np.linalg.norm(vertex_force, axis=-1)[:, np.newaxis] + EPSILON
        return vertex_force

    def calculate_face_normals(self):
        normals = np.zeros((MAX_VERTICES, 3), dtype=np.float32)

        # Get active faces
        active_faces = self.face_idx[self.face_idx != -1]
        if len(active_faces) == 0:
            return normals

        # Get triangle vertices using the same pattern as in draw()
        triangle_first_half_edges = self.face_half_edge[active_faces]
        triangle_half_edges = np.stack([
            triangle_first_half_edges,
            self.half_edge_next[triangle_first_half_edges],
            self.half_edge_next[self.half_edge_next[triangle_first_half_edges]],
        ], -1)
        triangle_vertices = self.half_edge_dest[triangle_half_edges]

        # Get vertex positions for all triangles
        triangle_positions = self.vertex_pos[triangle_vertices]  # shape: (n_faces, 3, 3)

        # Calculate face normals using cross product
        edge1 = triangle_positions[:, 1] - triangle_positions[:, 0]  # v2 - v1
        edge2 = triangle_positions[:, 2] - triangle_positions[:, 0]  # v3 - v1
        face_normals = np.cross(edge1, edge2)  # shape: (n_faces, 3)

        # Normalize face normals
        norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
        face_normals = np.where(norms > EPSILON, face_normals / norms, np.array([0.0, 0.0, 1.0]))

        # Add face normals to vertex normals
        for i in range(3):  # For each vertex in triangle
            np.add.at(normals, triangle_vertices[:, i], face_normals)

        # Normalize vertex normals
        vertex_norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = np.where(vertex_norms > EPSILON, normals / vertex_norms, np.array([0.0, 0.0, 1.0]))

        return normals

    def draw_pygame(self, screen):
        screen.fill((0, 0, 0))
        
        # Draw connections
        _, edge_pos = self.get_edge_positions()
        for i in range(edge_pos.shape[1]):
            link_start = tuple(map(float, edge_pos[0, i]))
            link_end = tuple(map(float, edge_pos[1, i]))
            pygame.draw.line(screen, (50, 50, 50), link_start, link_end, 2)
        
        # Draw cells
        active_cells = np.where(self.vertex_idx != -1)[0]
        cell_suitability = self.vertex_suitability[active_cells]
        active_cell_positions = self.vertex_pos[active_cells]
        for active_cell in active_cells:
            cell_position = self.vertex_pos[active_cell]
            cell_suitability = self.vertex_suitability[active_cell]
            pos = tuple(map(float, cell_position))
            pygame.draw.circle(screen, (int(255 * cell_suitability), int(255 * cell_suitability), int(255 * cell_suitability)), pos, 5)

        pygame.display.flip()

    def camera_matrix(self, width, height):
        return glm.ortho(0, width, height, 0, -1.0, 1.0)  

    def draw(self, screen):
        active_faces = self.face_idx[self.face_idx != -1]
        if len(active_faces) == 0:
            return

        # Calculate normals
        normals = self.calculate_face_normals()

        # Write ALL vertices to the buffer
        vertex_pos_3d = np.stack([self.vertex_pos[:, 0:1], self.vertex_pos[:, 1:2], np.zeros((self.vertex_pos.shape[0], 1))], axis=-1)
        all_vertices = np.ascontiguousarray(vertex_pos_3d.reshape(-1), dtype='f4')
        self.vbo.write(all_vertices.tobytes())

        # Write ALL normals to the buffer
        all_normals = np.ascontiguousarray(normals.reshape(-1), dtype='f4')
        self.nbo.write(all_normals.tobytes())

        # Get all half-edges that have faces (not boundary half-edges with face == -1)
        all_half_edges_with_faces = self.half_edge_idx[(self.half_edge_idx != -1) & (self.half_edge_face != -1)]

        # Build triangles for all half-edges with faces
        triangle_half_edges = np.stack([
            all_half_edges_with_faces,
            self.half_edge_next[all_half_edges_with_faces],
            self.half_edge_next[self.half_edge_next[all_half_edges_with_faces]],
        ], -1)
        triangle_vertices = self.half_edge_dest[triangle_half_edges]

        # Flatten to get index array
        indices = np.ascontiguousarray(triangle_vertices.reshape(-1), dtype=np.uint32)
        self.ibo.write(indices.tobytes())

        camera = self.camera_matrix(screen.get_width(), screen.get_height())
        self.ctx.clear()
        self.ctx.enable(self.ctx.DEPTH_TEST)
        self.ctx.disable(self.ctx.CULL_FACE)

        # Set lighting uniforms
        light_pos = np.array([screen.get_width()/2, screen.get_height()/4, 200.0], dtype='f4')
        light_color = np.array([0.8, 0.8, 0.8], dtype='f4')
        ambient_color = np.array([0.2, 0.2, 0.2], dtype='f4')

        self.program['camera'].write(camera)
        self.program['light_pos'].write(light_pos.tobytes())
        self.program['light_color'].write(light_color.tobytes())
        self.program['ambient_color'].write(ambient_color.tobytes())

        # Render using the number of indices, not vertices
        if self.wireframe_mode:
            self.ctx.wireframe = True
            self.vao.render(mode=moderngl.TRIANGLES, vertices=len(indices))
            self.ctx.wireframe = False
        else:
            self.vao.render(mode=moderngl.TRIANGLES, vertices=len(indices))
        pygame.display.flip()


def main():
    np.random.seed(42)

    width, height = 1400, 1400

    mesh = Mesh()

    mesh.make_first_triangle(width, height)
    # mesh.test()
    mesh.add_external_triangle(0, 2)

    clock = pygame.time.Clock()
    pygame.init()
    screen = pygame.display.set_mode((width, height), flags=pygame.OPENGL | pygame.DOUBLEBUF, vsync=True)

    mesh.init_shader()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w:
                    mesh.wireframe_mode = not mesh.wireframe_mode
                    print(f"Wireframe mode: {'ON' if mesh.wireframe_mode else 'OFF'}")
        
        mesh.update()
        mesh.draw(screen)
        clock.tick(60)

    pygame.quit()

if __name__ == '__main__':
    main()