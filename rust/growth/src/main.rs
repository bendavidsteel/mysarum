
use nannou::prelude::*;

const E: f32 = 2.718281828459045;
const MAX_PARTICLES: usize = 200;

fn main() {
    nannou::app(model).update(update).run();
}

fn growth(x: f32, mu: f32, sigma: f32) -> f32 {
    E.powf(-0.5 * pow((x - mu) / sigma, 2))
}

struct Node {
    pub x: f32,
    pub y: f32,
    state: f32,
    radius: f32,   // Radius of impact
    ramp: f32,     // Influences the shape of the function
    strength: f32, // Strength: positive value attracts, negative value repels
    pub damping: f32,
    pub velocity: Vec2,
    max_velocity: f32,
    u: f32,
}

impl Node {
    fn new(x: f32, y: f32, state: f32) -> Self {
        Node {
            x,
            y,
            state: state,
            radius: 100.0,
            ramp: 1.0,
            strength: -5.0,
            damping: 0.5,
            velocity: vec2(0.0, 0.0),
            max_velocity: 10.0,
            u: 0.0,
        }
    }

    fn update(&mut self, min_x: f32, max_x: f32, min_y: f32, max_y: f32, state_mu: f32, state_sigma: f32, dt: f32) {
        self.velocity = self.velocity.clamp_length_max(self.max_velocity);

        self.x += self.velocity.x;
        self.y += self.velocity.y;

        if self.x < min_x {
            self.x = min_x - (self.x - min_x);
            self.velocity.x = -self.velocity.x;
        }
        if self.x > max_x {
            self.x = max_x - (self.x - max_x);
            self.velocity.x = -self.velocity.x;
        }

        if self.y < min_y {
            self.y = min_y + (self.y - min_y);
            self.velocity.y = -self.velocity.y;
        }
        if self.y > max_y {
            self.y = max_y + (self.y - max_y);
            self.velocity.y = -self.velocity.y;
        }

        self.velocity *= 1.0 - self.damping;

        self.state += growth(self.u, state_mu, state_sigma) * dt;
        self.state = self.state.clamp(0.0, 1.0);
    }
    
    fn should_split(&self, growth_mu: f32, growth_sigma: f32, growth_prob: f32) -> bool {
        random_f32() > growth(self.u, growth_mu, growth_sigma) && random_f32() > growth_prob
    }

    fn color(&self) -> Color {
        let hue: f32 = map_range(self.state, 0.0, 1.0, 180.0, 250.0);
        Color::hsva(hue, 1.0, 1.0, 1.0)
    }
}

struct Model {
    nodes: Vec<Node>,
    spring_connections: Vec<(usize, usize)>,
    node_radius: f32,
    node_count: usize,
    state_mu: f32,
    state_sigma: f32,
    growth_mu: f32,
    growth_sigma: f32,
    growth_prob: f32,
    dt: f32,
    spring_length: f32,
}

fn model(app: &App) -> Model {
    app.new_window()
        .size(1000, 1000)
        .view(view)
        .key_pressed(key_pressed)
        .build();

    let node_radius = 4.0;
    let node_count = 10;
    let nodes = create_nodes(node_count, node_radius, app.window_rect());
    let spring_connections = create_connections(node_count);

    Model {
        nodes,
        spring_connections,
        node_radius,
        node_count,
        state_mu: random_f32(),
        state_sigma: random_f32(),
        growth_mu: random_f32(),
        growth_sigma: random_f32(),
        growth_prob: 0.99 + 0.01 * random_f32(),
        dt: random_f32() * 0.01,
        spring_length: 5.0,
    }
}

fn create_connections(node_count: usize) -> Vec<(usize, usize)> {
    let vector_of_tuples: Vec<(usize, usize)> = vec![
        (0, 1),
        (1, 2),
        (1, 3),
        (2, 3),
        (2, 4),
        (3, 5),
        (4, 5),
        (4, 6),
        (5, 7),
        (6, 7),
        (6, 8),
        (7, 9),
        (8, 9),
        (0, 9),
        (0, 8)
    ];
    vector_of_tuples
}

fn create_nodes(node_count: usize, node_radius: f32, win: geom::Rect) -> Vec<Node> {
    (0..node_count)
        .map(|_| {
            Node::new(
                random_range(-200.0, 200.0),
                random_range(-200.0, 200.0),
                random_f32(),
            )
        })
        .collect()
}

fn attract_nodes(nodes: &mut Vec<Node>, target: usize) {
    for other in 0..nodes.len() {
        // Continue from the top when node is itself
        if other == target {
            continue;
        }
        let df = attract(&nodes[target], &nodes[other]);
        nodes[other].velocity += df;
    }
}

fn attract(current_node: &Node, other_node: &Node) -> Vec2 {
    let current_node_vector = vec2(current_node.x, current_node.y);
    let other_node_vector = vec2(other_node.x, other_node.y);
    let d = current_node_vector.distance(other_node_vector);

    if d > 0.0 && d < current_node.radius {
        let s = (d / current_node.radius).powf(1.0 / current_node.ramp);
        let f = s * 9.0 * current_node.strength * (1.0 / (s + 1.0) + ((s - 3.0) / 4.0)) / d;
        let mut df = current_node_vector - other_node_vector;
        df *= f;
        df
    } else {
        vec2(0.0, 0.0)
    }
}

// ------ apply forces on spring and attached nodes ------
fn spring(nodes: &mut Vec<Node>, spring_connection: (usize, usize), length: f32) {
    let stiffness = 1.0;
    let damping = 0.9;

    let (from_node, to_node) = spring_connection;
    let mut diff =
        vec2(nodes[to_node].x, nodes[to_node].y) - vec2(nodes[from_node].x, nodes[from_node].y);
    diff = diff.normalize();
    diff *= length;
    let target = vec2(nodes[from_node].x, nodes[from_node].y) + diff;

    let mut force = target - vec2(nodes[to_node].x, nodes[to_node].y);
    force *= 0.5;
    force *= stiffness;
    force *= 1.0 - damping;

    nodes[to_node].velocity += force;
    force *= -1.0;
    nodes[from_node].velocity += force;
}

fn message_pass(nodes: &mut Vec<Node>, spring_connection: (usize, usize)) {
    let (from_node, to_node) = spring_connection;
    nodes[from_node].u += nodes[to_node].state;
    nodes[to_node].u += nodes[from_node].state;
}

fn update(app: &App, model: &mut Model) {
    for i in 0..model.nodes.len() {
        // Let all nodes repel each other
        attract_nodes(&mut model.nodes, i);
    }

    for connection in model.spring_connections.iter() {
        // apply spring forces
        spring(&mut model.nodes, *connection, model.spring_length);
        message_pass(&mut model.nodes, *connection);
    }

    let win = app.window_rect();

    for i in 0..model.nodes.len() {
        // Apply velocity vector and update position
        let node_radius = model.nodes[i].radius;
        model.nodes[i].update(
            win.left() + node_radius,
            win.right() - node_radius,
            win.top() - node_radius,
            win.bottom() + node_radius,
            model.state_mu,
            model.state_sigma,
            model.dt
        );
    }

    for i in 0..model.nodes.len() {
        if model.nodes.len() < MAX_PARTICLES && model.nodes[i].should_split(model.growth_mu, model.growth_sigma, model.growth_prob) {
            let new_node_a = Node::new(
                model.nodes[i].x + random_range(-10.0, 10.0),
                model.nodes[i].y + random_range(-10.0, 10.0),
                model.nodes[i].state,
            );
            let new_node_b = Node::new(
                model.nodes[i].x + random_range(-10.0, 10.0),
                model.nodes[i].y + random_range(-10.0, 10.0),
                model.nodes[i].state,
            );            
            model.nodes.push(new_node_a);
            model.nodes.push(new_node_b);

            // add connections to new nodes
            let node_a_idx = model.nodes.len() - 2;
            let node_b_idx = model.nodes.len() - 1;
            model.spring_connections.push((i, node_a_idx));
            model.spring_connections.push((i, node_b_idx));
            model.spring_connections.push((node_a_idx, node_b_idx));

            // edit connections to old node
            let mut connection_counter = 0;
            for connection in model.spring_connections.iter_mut() {
                if connection.0 == i {
                    if connection_counter == 1 {
                        connection.0 = node_a_idx;
                    } else if connection_counter == 2 {
                        connection.0 = node_b_idx;
                    }
                    connection_counter += 1;
                } else if connection.1 == i {
                    if connection_counter == 1 {
                        connection.1 = node_a_idx;
                    } else if connection_counter == 2 {
                        connection.1 = node_b_idx;
                    }
                    connection_counter += 1;
                }
                
                if connection_counter > 2 {
                    break;
                }
            }
        }
    }
}

fn view(app: &App, model: &Model) {
    // Begin drawing
    let draw = app.draw();
    draw.background().color(BLACK);

    model.spring_connections.iter().for_each(|connection| {
        // draw spring
        let (to, from) = *connection;
        draw.line()
            .start(pt2(model.nodes[from].x, model.nodes[from].y))
            .end(pt2(model.nodes[to].x, model.nodes[to].y))
            .stroke_weight(2.0)
            .srgb(0.0, 0.5, 0.64);
    });

    model.nodes.iter().for_each(|node| {
        draw.ellipse()
            .x_y(node.x, node.y)
            .radius(model.node_radius)
            .color(node.color())
            .stroke(WHITE)
            .stroke_weight(2.0);
    });
}

fn key_pressed(app: &App, model: &mut Model, key: KeyCode) {
    if key == KeyCode::KeyS {
        app.main_window()
            .save_screenshot(app.exe_name().unwrap() + ".png");
    }
    if key == KeyCode::KeyR {
        randomize_and_reset(model, app);
    }
}

fn randomize_and_reset(model: &mut Model, app: &App) {
    model.nodes = create_nodes(model.node_count, model.node_radius, app.window_rect());
    model.spring_connections = create_connections(model.node_count);

    model.state_mu = random_f32();
    model.state_sigma = random_f32();
    model.growth_mu = random_f32();
    model.growth_sigma = random_f32();
    model.growth_prob = 0.9 + 0.1 * random_f32();
    model.dt = random_f32() * 0.1;
}
