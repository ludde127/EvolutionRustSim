#![deny(clippy::all)]
#![forbid(unsafe_code)]

mod life;
mod mapped_pair;
//mod mapped_pair;

use std::cmp::min;
use std::time::{Instant};
use std::ops::{Add, Div, Mul, Range, Sub};
use log::error;
use pixels::{Error, Pixels, SurfaceTexture};
use winit::dpi::{LogicalSize};
use winit::event::{Event, VirtualKeyCode};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;
use winit_input_helper::WinitInputHelper;
use std::collections::{HashMap, HashSet};
use std::fmt;

const WIDTH: u32 = 1200;
const HEIGHT: u32 = 800;
use rand::Rng;
use crate::mapped_pair::MappedPair;

const WIDTH_AS_F64: f64 = WIDTH as f64;
const HEIGHT_AS_F64: f64 = HEIGHT as f64;
const WIDTH_HEIGHT: (f64, f64) = (WIDTH_AS_F64, HEIGHT_AS_F64);

const DENSITY: f64 = 997.0; // Kg/m3. waters density
const DRAG_COEFFICIENT: f64 = 0.0;
const PI: f64 = std::f64::consts::PI;
const NUMBER_OF_NON_PLAYER_PARTICLES: u8 = 15;

const TELEPORT_SHORT_DISTANCE_ON_IMPACT: bool = false;

struct RGBA {
    r: u8,
    g: u8,
    b: u8,
    a: u8,
}

struct Pixel {
    x: u16,
    y: u16,
    color: RGBA,
}

#[derive(Debug, Copy, Clone, PartialEq)]
struct Material {
    density: f64,
    elasticity: f64,
}

impl Material {
    fn default() -> Self {Material{density: 20.0, elasticity: 0.01}}

    fn random() -> Self {Material{
        density: random_in_bound(2..20),
        elasticity: random_in_bound(3..10)/10.0
    }}
}

trait PlayableArea {
    fn contains(&self, x: &f64, y: &f64) -> bool;
    fn collision_with_shape(&self, shape: &Shape) -> bool;
    fn collision_modelling(&self, particle: &mut Particle);
    fn x_contains(&self, x: &f64) -> bool;
    fn y_contains(&self, y: &f64) -> bool;
    fn corners_outside_area(&self, particle: &Particle) -> Vec<(bool, bool)>;
    //fn corners(&self) -> Vec<(f64)>;
}

struct BasicBox {
    x0: i32,
    y0: i32,
    width: i32,
    height: i32,
}

impl BasicBox {
    fn new() -> Self {
        Self {
            x0: 0,
            y0: 0,
            width: WIDTH as i32,
            height: HEIGHT as i32,
        }
    }
}

impl PlayableArea for BasicBox {
    fn contains(&self, x: &f64, y: &f64) -> bool {
        self.x_contains(x) && self.y_contains(y)
    }

    fn collision_with_shape(&self, shape: &Shape) -> bool {
        // Simple check for corners, Should work for the BasicBox.
        match shape {
            Shape::Circle(shape) =>
                shape.corners().iter().any(|(c0, c1)| !self.contains(c0, c1)),

            Shape::Rectangle(shape) =>
                shape.corners().iter().any(|(c0, c1)| !self.contains(c0, c1)),
        }
    }

    fn collision_modelling(&self, particle: &mut Particle) {
        let corner_outside_bounds = self.corners_outside_area(particle);

        if corner_outside_bounds.iter().any(|(x, y)| *x || *y) {
            // They collided, handle it!
            let x_collision = corner_outside_bounds.iter().any(|(x, y)| *x);
            let y_collision = corner_outside_bounds.iter().any(|(x, y)| *y);



            let mut current = particle.velocity;
            if !x_collision {
                current.set_y(-1.0*current.y);
            }
            if !y_collision {
                current.set_x(-1.0*current.x);
            }

            if !x_collision && !y_collision {
                let are_all_corners_outside =
                    corner_outside_bounds.iter().all(|(x, y)| *x && *y);
                if are_all_corners_outside {
                    particle.randomize();
                    return;
                }
            }

            particle.set_velocity(current);
            if TELEPORT_SHORT_DISTANCE_ON_IMPACT {
                // Teleport 3 blocks away.
                let mut tp = particle.geometry.center();

                if ((self.x0 as f64) - tp.x).abs() < (((self.x0 + self.width) as f64) - tp.x).abs() {
                    tp.set_x(tp.x + 1.0);
                } else {
                    tp.set_x(tp.x - 1.0);
                }

                if ((self.y0 as f64) - tp.y).abs() < (((self.y0 + self.height) as f64) - tp.y).abs() {
                    tp.set_y(tp.y + 1.0);
                } else {
                    tp.set_y(tp.y - 1.0);
                }
                particle.set_center(tp); // Teleport it out a bit
            }
            //println!("Debugg {}", x_collision);

        }
    }

    fn x_contains(&self, x: &f64) -> bool {
        let x = (*x) as i32;
        (x >= self.x0 && x <= (self.x0 + self.width))
    }

    fn y_contains(&self, y: &f64) -> bool {
        let y = (*y) as i32;
        (y >= self.y0 && y <= (self.y0 + self.height))
    }

    fn corners_outside_area(&self, particle: &Particle) -> Vec<(bool, bool)> {
        Vec::from_iter(particle.geometry.corners().iter().map(
            |(c0, c1)| (!self.x_contains(c0), !self.y_contains(c1))).into_iter())
    }
}

trait RemoveMultiple <T> {
    fn remove_multiple_borrowed_values(&mut self, indexes: &HashSet<usize>) -> Vec<&T>;
}

impl <T> RemoveMultiple <T> for Vec<T> {
    fn remove_multiple_borrowed_values(&mut self, indexes: &HashSet<usize>) -> Vec<&T> {
        assert!(indexes.len() <= self.len(),
                "Indexes to remove must be fewer than the existing indexes.");
        let mut new_vec:Vec<&T> = Vec::with_capacity(self.len()-indexes.len());

        for index in 0..(new_vec.len()-1) {
            if !indexes.contains(&index) {
                new_vec.push(self.get(index).unwrap());
            }
        }
        new_vec
    }
}

trait Geometry: std::fmt::Debug {
    fn contains(&self, x: &f64, y: &f64) -> bool;
    fn area(&self) -> f64;
    fn center(&self) -> Pair;
    fn _move(&mut self, x: f64, y: f64);
    fn position(&self) -> Pair;
    fn collided_with(&self, other: &Shape) -> bool;
    fn shape(&self) -> Shape;
    fn drag_area(&self) -> f64;
    fn corners(&self) -> Vec<(f64, f64)>;
}

#[derive(Debug, Clone, PartialEq)]
enum Shape {
    Rectangle(Rectangle),
    Circle(Circle),
}

fn circle_rectangle_intersect(circle: &Circle, rectangle: &Rectangle) -> bool {
    let center_dist = (circle.center()-rectangle.center()).abs();

    if center_dist.x > (rectangle.length/2.0 + circle.diameter/2.0) ||
        (center_dist.y > (rectangle.height/2.0 + circle.diameter/2.0)) {
        false
    } else if center_dist.x <= rectangle.length/2.0 || center_dist.y <= rectangle.height/2.0 {
        true
    } else {
        (center_dist-Pair::new(rectangle.length/2.0, rectangle.height/2.0)).pythagoras()
            <= (circle.diameter/2.0).powf(2.0)
    }
}

fn collided(shape_one: &Shape, shape_two: &Shape) -> bool {
    match shape_one {
        Shape::Rectangle(rectangle1) => {
            match shape_two {
                Shape::Rectangle(rectangle2) => {
                    rectangle1.corners().iter().any(|c| rectangle2.contains(&c.0, &c.1))
                }
                Shape::Circle(circle2) => {
                    circle_rectangle_intersect(circle2, rectangle1)
                }
            }
        },
        Shape::Circle(circle1) => {
            match shape_two {
                Shape::Rectangle(rectangle2) => {
                    circle_rectangle_intersect(circle1, rectangle2)
                },
                Shape::Circle(circle2) => {
                    circle1.center().distance(circle2.center()) <= circle1.diameter/2.0+circle2.diameter/2.0
                },
            }
        }
    }
}



#[derive(Debug, Copy, Clone, PartialEq)]
struct Rectangle {
    x0: f64,
    y0: f64,
    length: f64,
    height: f64,
}

impl Rectangle {
    fn new(x0: f64, y0: f64, length: f64, height: f64) -> Self {
        Self {
            x0,
            y0,
            length,
            height
        }
    }

    fn new_square(x: f64, y: f64, side: f64) -> Self {
        Self {
            x0: x,
            y0: y,
            length: side,
            height: side,
        }
    }
    
    
    fn __x_in_geometry(&self, x: &f64) -> bool {
        &self.x0 <= x && x <= &(self.x0+self.length)

    }

    fn __y_in_geometry(&self, y: &f64) -> bool {
        &self.y0 <= y && y <= &(self.y0+self.height)
    }
}

impl Geometry for Rectangle {
    fn contains(&self, x: &f64, y: &f64) -> bool {
        // First check should be faster than lower checks.
        self.__x_in_geometry(x) && self.__y_in_geometry(y)
    }
    
    fn area(&self) -> f64 {((self.x0+self.length)-self.x0)*((self.y0+self.height)-self.y0)}
    
    fn center(&self) -> Pair {
        Pair::new((((self.x0+self.length) - self.x0) as f64)/2.0, (((self.y0+self.height) - self.y0) as f64)/2.0)
    }

    fn _move(&mut self, x: f64, y: f64) {
        self.x0 = x;
        self.y0 = y;
    }

    fn position(&self) -> Pair {
        Pair::new(self.x0, self.y0)
    }

    fn collided_with(&self, other: &Shape) -> bool {
        collided(&self.shape(), other)
    }

    fn shape(&self) -> Shape {
        Shape::Rectangle(*self)
    }

    fn drag_area(&self) -> f64 {
        (self.length+self.height)*0.5
    }

    fn corners(&self) -> Vec<(f64, f64)> {
        let left_upper = (self.x0, self.y0);
        let right_lower = ((self.x0+self.length), (self.y0+self.height));
        let left_lower = (self.x0, (self.y0+self.height));
        let right_upper = ((self.x0+self.length), self.y0);
        vec![left_upper, right_lower, left_lower, right_upper]
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
struct Circle {
    x: f64, // Center
    y: f64, // Center
    diameter: f64,
}

impl Circle {
    fn new(x: f64, y: f64, diameter: f64) -> Self {
        Self { x, y, diameter}
    }
}

impl Geometry for Circle {
    fn contains(&self, x: &f64, y: &f64) -> bool {
        let given = Pair::new(*x, *y);
        let diff = self.center() - given;
        let normal =  diff.pythagoras()<=self.diameter/2.0; // True if they lay in same modulo plane and intersect
        normal
    }

    fn area(&self) -> f64 {
        (self.diameter/2.0).powf(2.0)*PI
    }

    fn center(&self) -> Pair {
        Pair::new(self.x, self.y)
    }

    fn _move(&mut self, x: f64, y: f64) {
        self.x = x;
        self.y = y;
    }

    fn position(&self) -> Pair {
        self.center()
    }

    fn collided_with(&self, other: &Shape) -> bool {
        collided(&self.shape(), other)
    }

    fn shape(&self) -> Shape {
        Shape::Circle(*self)
    }

    fn drag_area(&self) -> f64 {
        self.diameter
    }

    fn corners(&self) -> Vec<(f64, f64)> {
        /// Cornors of the smallest square which can fit the circle.
        let x0 = self.x - self.diameter/2.0;
        let x1 = self.x + self.diameter/2.0;
        let y0 = self.y - self.diameter/2.0;
        let y1 = self.y + self.diameter/2.0;

        let left_upper = (x0, y0);
        let right_lower = (x1, y1);
        let left_lower = (x0, y1);
        let right_upper = (x1, y0);
        vec![left_upper, right_lower, left_lower, right_upper]
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
struct Pair {
    x: f64,
    y: f64,
}

fn random_in_bound(range: Range<i32>) -> f64 {
    let mut rng = rand::thread_rng();
    rng.gen::<f64>() * ((range.end - range.start) as f64) + (range.start as f64)
}

impl Pair {
    fn new(x: f64, y: f64) -> Self {
        Self {
            x,
            y
        }
    }

    fn random_in_bound(x_range: Range<i32>, y_range: Range<i32>) -> Self {
        Pair::new(random_in_bound(x_range), random_in_bound(y_range))
    }

    fn width_height() -> Self {
        Pair::new(WIDTH_AS_F64, HEIGHT_AS_F64)
    }

    fn to_tuple(&self) -> (f64, f64) {
        (self.x, self.y)
    }

    fn zeros() -> Self {
        Pair::new(0.0, 0.0)
    }

    fn value(val: f64) -> Self {
        Pair::new(val, val)
    }
    
    fn set_x(&mut self, x: f64) {
        self.x = x;
    }

    fn set_y(&mut self, y: f64) {
        self.y = y;
    }

    fn abs(&mut self) -> Self {self.x = self.x.abs(); self.y=self.y.abs(); self.clone()}

    fn to_dir(&self) -> Self {
        let mut temp = Pair::zeros();
        if self.x > 0.0 {
            temp.set_x(1.0);
        } else if self.x < 0.0 {
            temp.set_x(-1.0);
        }

        if self.y > 0.0 {
            temp.set_y(1.0);
        } else if self.y < 0.0 {
            temp.set_y(-1.0);
        }
        
        temp
    }

    fn modulo(&self, b: Pair) -> Self {
        let mut x = self.x%b.x;
        let mut y = self.y%b.y;
        if x < 0.0 {x = x + b.x};
        if y < 0.0 {y = y + b.y};
        Pair::new(x, y) // This should have the behaviour of normal modulo operations in math.
    }

    fn modulo_window(&self) -> Self { // Shorthand for fitting to the window as this is very common.
        self.modulo(Pair::width_height())
    }

    fn squared(&self) -> Self {Pair::new(self.x*self.x, self.y*self.y)}

    fn distance(&self, other: Pair) -> f64 {
        ((self.x-other.x).powf(2.0)+(self.y-other.y).powf(2.0)).sqrt()
    }

    fn pythagoras(&self) -> f64 {
        (self.x.powf(2.0) + self.y.powf(2.0)).sqrt()
    }

    fn cross_prod(&self, other: Pair) -> f64 {
        self.y*other.x - self.x*other.y
    }

    fn dot_prod(&self, other: Pair) -> f64 {
        self.x*other.y + self.y*other.x
    }

    fn inverse(&self) -> Self {
        // Inverses the vector pair.
        Self {
            x: self.x * -1.0,
            y: self.y * -1.0,
        }
    }
}

impl fmt::Display for Pair {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "x: {}, y: {}", self.x, self.y)
    }
}

fn value(val: f64) -> Pair  {Pair::value(val)} // Kinda dumb way to handle numbers but what ever.

impl Add for Pair {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}

impl Mul for Pair {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x * rhs.x,
            y: self.y * rhs.y,
        }
    }
}

impl Mul<f64> for Pair {
    type Output = Pair;

    fn mul(self, rhs: f64) -> Pair {
        Self {
            x: self.x * rhs,
            y: self.y * rhs,
        }
    }
}

impl Mul<Pair> for f64 {
    type Output = Pair;

    fn mul(self, rhs: Pair) -> Pair {
        rhs * self
    }
}

impl Div for Pair {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        // Divides self by the rhs Pair coordinate wise
        Self {
            x: self.x / rhs.x,
            y: self.y / rhs.y,
        }
    }
}

impl Sub for Pair {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self { x: self.x - rhs.x, y: self.y - rhs.y }
    }
}

#[derive(Debug)]
struct Controller {
    is_player_controlled: bool, // Usize is here to later allow more players than one
    id: Option<usize>,
}

impl Controller {
    fn unset_player() -> Self {
        Self {is_player_controlled: false, id: None}
    }
    
    fn not_player(id: usize) -> Self {
        Self {is_player_controlled: false, id: Some(id)}
    }

    fn player(id: usize) -> Self {
        Self {is_player_controlled: true, id: Some(id)}
    }
    
    fn get_id(&self) -> usize {
        self.id.unwrap()
    }
}

#[derive(Debug)]
struct Particle {
    geometry: Box<dyn Geometry>,
    velocity: Pair,
    acceleration: Pair,
    material: Material,
    dt: Instant,
    controller: Controller,
    paused: bool,
    was_paused: bool,
}
impl Particle {
    fn new(geometry: Box<dyn Geometry>, velocity: Pair, acceleration: Pair, material: Material) -> Self {
        Self {
            geometry,
            velocity,
            acceleration,
            material,
            dt: Instant::now(),
            controller: Controller::unset_player(),
            paused: false,
            was_paused: false,
        }
    }

    fn new_square(position: Pair, velocity: Pair, acceleration: Pair, material: Material, height: f64) -> Self {
        Self {
            geometry: Box::new(Rectangle::new(position.x, position.x + height,
                                                      position.y, position.y + height)),
            velocity,
            acceleration,
            material,
            dt: Instant::now(),
            controller: Controller::unset_player(),
            paused: false,
            was_paused: false,
        }
    }
    
    fn new_still_rectangle(x: f64, y: f64, length: f64, height: f64) -> Self {
        Self {
            geometry: Box::new(Rectangle::new(x, x + length,
                                                      y, y + height)),
            velocity: Pair::zeros(),
            acceleration: Pair::zeros(),
            material: Material::default(),
            controller: Controller::unset_player(),
            dt: Instant::now(),
            paused: false,
            was_paused: false
        }
    }
    
    fn new_still_square(x: f64, y:f64, side: f64) -> Self {
        Particle::new_square(Pair::new(x, y), Pair::zeros(),
                             Pair::zeros(), Material::default(), side)
    }

    fn new_still_circle(x: f64, y: f64, diameter: f64) -> Self {
        Self {
            geometry: Box::new(Circle::new(x, y, diameter)),
            velocity: Pair::zeros(),
            acceleration: Pair::zeros(),
            material: Material::default(),
            dt: Instant::now(),
            paused: false,
            was_paused: false,
            controller: Controller::unset_player(),
        }
    }

    fn new_random_circle() -> Self {
        let diameter = random_in_bound(10..55);
        let rand_pos = Pair::random_in_bound(
            ((diameter*1.1) as i32)..((WIDTH_AS_F64-(diameter*1.1)) as i32),
            ((diameter*1.1) as i32)..((HEIGHT_AS_F64-(diameter*1.1)) as i32)
        );
        Self {
            geometry: Box::new(Circle::new(rand_pos.x, rand_pos.y, diameter)),
            velocity: Pair::random_in_bound(-13..13, -13..13),
            acceleration: Pair::zeros(),
            material: Material::random(),
            dt: Instant::now(),
            paused: false,
            was_paused: false,
            controller: Controller::unset_player(),
        }
    }

    fn set_as_player(&mut self, id: usize) {self.controller = Controller::player(id)}

    fn drag_force(&self) -> Pair {
        value(0.5) * self.velocity.to_dir() * value((-1.0)) *
            self.velocity * self.velocity * value(self.geometry.drag_area() * DRAG_COEFFICIENT)
    }

    fn update(&mut self) {
        assert!(!self.paused, "Tried to update while paused");
        if !self.was_paused {
            let dt_pair = value(self.dt.elapsed().as_secs_f64());

            self.apply_force(self.drag_force());
            //println!("{}", self.position);
            self.velocity = self.velocity + self.acceleration * dt_pair;

            // Modulo position to wrap around window.
            let pos = self.geometry.position() + self.velocity * dt_pair;
            self.geometry._move(pos.x, pos.y);
            self.acceleration = Pair::zeros();
        }
        self.was_paused = false;
        self.dt = Instant::now(); // Reset the dt
    }

    fn occupies(&self, x: &f64, y: &f64) -> bool {
        self.geometry.contains(x, y)
    }

    fn apply_force(&mut self, force: Pair){
        self.acceleration = self.acceleration + force/value(self.mass());
    }

    fn set_velocity(&mut self, velocity: Pair) {
        self.velocity = velocity;
        self.acceleration = Pair::zeros(); // This is needed as the velocity was forcefully changed
    }

    fn toggle_pause(&mut self){
        self.paused = !self.paused;
        self.was_paused = !self.paused;
    }

    fn mass(&self) -> f64 {self.material.density * self.geometry.area()}

    fn collides_with(&self, other: &Particle) -> bool {
        self.geometry.collided_with(&other.geometry.shape())
    }

    fn randomize(&mut self) {
        let random_p = Particle::new_random_circle();
        self.velocity = random_p.velocity;
        self.material = random_p.material;
        self.geometry = random_p.geometry;
    }

    fn set_center(&mut self, position: Pair) {
        self.geometry._move(position.x, position.y);
    }
}

fn assign_to_quadrants(shape: &Box<dyn Geometry>, median_diameter: f64) -> HashSet<(u8, u8)> {
    // Small quadrants seams to work best
    let quadrant_width = 0.5*WIDTH_AS_F64/median_diameter;
    let quadrant_height = 0.5*HEIGHT_AS_F64/median_diameter;
    let hs: HashSet<(u8, u8)> = HashSet::from_iter(shape.corners().iter()
        .map(|cor| (((quadrant_width*(cor.0)/WIDTH_AS_F64) as u8),
                    ((quadrant_height*(cor.1)/HEIGHT_AS_F64) as u8)))
        );
    hs
}

struct ObjectCoordinator {
    objects: Vec<Particle>,
    playable_area: Box<dyn PlayableArea>,
    last_collisions: HashSet<MappedPair<usize>>, // Holds last checks collision
    max_used_id: u32,
    paused: bool,
}

impl ObjectCoordinator {
    fn new() -> Self {
        Self {
            objects: vec![],
            paused: false,
            playable_area: Box::new(BasicBox::new()),
            last_collisions: HashSet::new(),
            max_used_id: 0
        }
    }

    /// Draw the `World` state to the frame buffer.
    ///
    /// Assumes the default texture format: `wgpu::TextureFormat::Rgba8UnormSrgb`
    fn draw(&self, frame: &mut [u8]) {
        for (i, pixel) in frame.chunks_exact_mut(4).enumerate() {
            let i = i as u32;
            let x = (i % WIDTH) as f64;
            let y = (i / WIDTH) as f64;
            if self.any_occupies(&x, &y) {
                pixel.copy_from_slice(&[0x5e, 0x48, 0xe8, 0xff]);
            } else {
                pixel.copy_from_slice(&[0x48, 0xb2, 0xe8, 0xff]);
            }
        }
    }

    fn any_occupies(&self, x: &f64, y: &f64) -> bool {
        self.objects.iter().any(|ps| ps.occupies(x, y))
    }

    fn collision_handling(&mut self) {
        // TODO MAKE THIS MORE EFFICIENT
        if &self.objects.len() > &1 {
            let mut valid_collisions: HashSet<MappedPair<usize>> = HashSet::new();
            let mut checked_pairs: HashSet<MappedPair<usize>> = HashSet::new();
            let mut collisions: HashSet<MappedPair<usize>> = HashSet::new();

            // Create the quadrants with the index of the particles in the storage vector instead.
            let quadrant_assignment: HashMap<usize, HashSet<(u8, u8)>> =
                HashMap::from_iter(self.objects.iter().enumerate()
                    .map(|(n,p)| (n, assign_to_quadrants(&p.geometry,
                                                         self.mean_diameter()))));
            for (n, particle) in self.objects.iter().enumerate() {
                for (n2, other) in self.objects.iter().enumerate() {
                    if n != n2 && !checked_pairs.contains(&MappedPair::new(n, n2))
                        && !quadrant_assignment[&n].
                            is_disjoint(&quadrant_assignment[&n2]){
                        if particle.collides_with(other) {
                            collisions.insert(MappedPair::new(n, n2));
                        }
                        checked_pairs.insert(MappedPair::new(n, n2));
                    }
                }
            }
            // Handle the collisions between particles.
            for pair in &collisions {
                if !self.last_collisions.contains(&pair){
                    valid_collisions.insert(*pair);
                }

            }
            self.last_collisions = collisions;
            self.handle_collision_between(&valid_collisions);
        // Detect and model impact between playable area and particle.

        self.objects.iter_mut().for_each(|particle|
            self.playable_area.collision_modelling(particle));
        }
    }

    fn handle_collision_between(&mut self, validated_pairs: &HashSet<MappedPair<usize>>) {
        let mut to_add: Vec<Particle> = Vec::new();

        for pair in validated_pairs {

            let obj1 = self.objects.get(*pair.first());
            let obj2 = self.objects.get(*pair.second());
            if obj1.is_some() && obj2.is_some() {
                let obj1 = obj1.unwrap();
                let obj2 = obj2.unwrap();
                let v1 = obj1.velocity;
                let v2 = obj2.velocity;

                let m1 = obj1.mass();
                let m2 = obj2.mass();

                let obj1_pos = obj1.geometry.position();
                let obj2_pos = obj2.geometry.position();

                let sys_mass = &(m1 + m2);
                if obj1.geometry.contains(&obj2_pos.x, &obj2_pos.y) ||
                    obj2.geometry.contains(&obj1_pos.x, &obj1_pos.y) {
                    // Merge them!
                    let v_result = (m1 * v1 + m2 * v2) * (1.0 / sys_mass);


                    let was_player = obj1.controller.is_player_controlled ||
                        obj2.controller.is_player_controlled;

                    let mean_pos = (obj1_pos + obj2_pos) * 0.5;
                    // r**2*pi = A r**2=A/pi r = sqrt(A/pi)
                    let d3 = 2.0 * ((obj1.geometry.area() + obj2.geometry.area()) / PI).sqrt();

                    let mut new = Particle::new(
                        Box::new(Circle::new(mean_pos.x, mean_pos.y, d3)),
                        v_result, Pair::zeros(), obj1.material);


                    if was_player {
                        // If first was player controlled then p1 is set for new else p2, if both p1
                        if obj1.controller.is_player_controlled {
                            new.set_as_player(obj1.controller.get_id());
                        } else {
                            new.set_as_player(obj2.controller.get_id());
                        }
                    }

                    to_add.push(new);
                    self.objects.remove(*pair.first());
                    if pair.first() < pair.second() {
                        self.objects.remove(*pair.second() - 1);
                    } else {
                        self.objects.remove(*pair.second());
                    }

                } else {
                    let cr1 = 2.0 * obj1.material.elasticity /
                        (obj1.material.elasticity + obj2.material.elasticity);
                    let cr2 = 2.0 * obj2.material.elasticity /
                        (obj1.material.elasticity + obj2.material.elasticity);
                    let common_part = m1 * v1 + m2 * v2;
                    let res1 = (common_part + m2 * cr1 * (v2 - v1)) * (1.0 / sys_mass);
                    let res2 = (common_part + m1 * cr2 * (v1 - v2)) * (1.0 / sys_mass);

                    // Teleport a bit to try to mitigate the stickiness

                    {
                        let obj1 = self.objects.get_mut(*pair.first()).unwrap();
                        obj1.set_velocity(res1*3.0);
                        if TELEPORT_SHORT_DISTANCE_ON_IMPACT {
                            obj1.set_center(obj1.geometry.center()+res1.to_dir());
                        }
                    }
                    {
                        let obj2 = self.objects.get_mut(*pair.second()).unwrap();
                        obj2.set_velocity(res2*3.0);
                        if TELEPORT_SHORT_DISTANCE_ON_IMPACT {
                            obj2.set_center(obj2.geometry.center()+res2.to_dir());
                        }
                    }
                    //self.last_collisions
                }
            }
        }
        self.objects.extend(to_add);
    }

    fn precise_occupies(&self, x: &f64, y: &f64) -> Vec<&Particle> {
        let mut occupying = vec![];
        for p in &self.objects {
            let occ = p.occupies(x, y);
            if occ {
                occupying.push(p);
            }
        }
        occupying
    }

    fn add(&mut self, mut particle: Particle) {
        particle.controller.id = self.max_used_id as usize;
        self.max_used_id += 1;
        self.objects.push(particle);
    }
    fn add_will_work(&mut self, particle: &Particle) -> bool {
        if self.playable_area.collision_with_shape(&particle.geometry.shape()) {
            false
        } else {
            let coll = self.objects.iter().any(|p| p.collides_with(&particle));
            if !coll {
                true
            } else {
                false
            }
        }
    }

    fn mean_diameter(&self) -> f64 {
        let mut diameter_sum = 0.0;
        let mut diameter_count = 0.0;
        for obj in &self.objects {
            diameter_sum += obj.geometry.drag_area();
            diameter_count += 1.0;
        }
        diameter_sum/diameter_count
    }

    fn apply_force_to_player(&mut self, player_id: usize, force: Pair) {
        self.objects.iter_mut().filter(|p|
            p.controller.is_player_controlled &&
                p.controller.get_id() == player_id).
            for_each(|p2| p2.apply_force(force));
    }

    fn add_player(&mut self, mut particle: Particle) -> Option<usize>{
        if self.add_will_work(&particle) {
            particle.controller = Controller::player(self.max_used_id as usize);
            self.max_used_id += 1;
            Some((self.max_used_id-1) as usize)
        } else {
            None
        }

    }

    fn update(&mut self) {
        if !self.paused {
            self.objects.iter_mut().for_each(|p| p.update());
            self.collision_handling();
        }
    }

    fn toggle_pause(&mut self) {
        self.paused = !self.paused;
        self.objects.iter_mut().for_each(|p| p.toggle_pause());
    }
}

fn main() -> Result<(), Error> {
    env_logger::init();
    let event_loop = EventLoop::new();
    let mut input = WinitInputHelper::new();
    let window = {
        let size = LogicalSize::new(WIDTH as f64, HEIGHT as f64);
        WindowBuilder::new()
            .with_title("Evolution")
            .with_inner_size(size)
            .with_min_inner_size(size)
            .build(&event_loop)
            .unwrap()
    };

    let mut pixels = {
        let window_width = window.inner_size();
        let surface_texture = SurfaceTexture::new(window_width.width, window_width.height, &window);
        Pixels::new(WIDTH, HEIGHT, surface_texture)?
    };

    let mut state = ObjectCoordinator::new();
    let mut added = false;
    let mut player: Option<usize> = None;

    while player.is_none() {
        let particle = Particle::new_random_circle();
        player = state.add_player(particle);
    }

    for n in 0..NUMBER_OF_NON_PLAYER_PARTICLES {
        added = false;
        while !added {
            let particle = Particle::new_random_circle();
            if state.add_will_work(&particle) {
                state.add(particle);
                added = true;
            }
        }
    }

    event_loop.run(move |event, _, control_flow| {
        // Draw the current frame
        if let Event::RedrawRequested(_) = event {
            state.draw(pixels.get_frame());
            if pixels
                .render()
                .map_err(|e| error!("pixels.render() failed: {}", e))
                .is_err()
            {
                *control_flow = ControlFlow::Exit;
                return;
            }
        }

        // Handle input events
        if input.update(&event) {
            // Close events
            if input.key_pressed(VirtualKeyCode::Escape) || input.quit() {
                *control_flow = ControlFlow::Exit;
                return;
            } else if input.key_pressed(VirtualKeyCode::Space) {
                state.toggle_pause();
            } else if input.key_pressed(VirtualKeyCode::Key1) {
                // Decrease energy in system
                state.objects.iter_mut().for_each(|p| p.set_velocity(p.velocity*0.95));
            } else if input.key_pressed(VirtualKeyCode::Key2) {
                state.objects.iter_mut().for_each(|p| p.set_velocity(p.velocity*1.05));
            } else if input.key_pressed(VirtualKeyCode::S) {
                state.add(Particle::new_random_circle());
            }

            if player.is_none() || state.paused {
            } else if input.key_held(VirtualKeyCode::Up) {
                state.apply_force_to_player(player.unwrap(), Pair::new(0.0, -10000000.0))
            } else if input.key_held(VirtualKeyCode::Down) {
                state.apply_force_to_player(player.unwrap(), Pair::new(0.0, 10000000.0))}
            else if input.key_held(VirtualKeyCode::Left) {
                state.apply_force_to_player(player.unwrap(), Pair::new(-10000000.0, 0.0))}
            else if input.key_held(VirtualKeyCode::Right) {
                state.apply_force_to_player(player.unwrap(), Pair::new(10000000.0, 0.0))}



            // Rewidth the window
            if let Some(width) = input.window_resized() {
                pixels.resize_surface(width.width, width.height);
            }

            // Update internal state and request a redraw

            state.update();
            window.request_redraw();
        }
    });
}
