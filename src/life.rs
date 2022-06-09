use std::collections::HashMap;

trait Sensor {

}

struct Sensors {

}

trait Reaction {

}

struct Reactions {

}

struct Neuron {

}

enum BrainParts {
    Sensors,
    Reactions,
    Neuron,
}

struct HiddenLayer {

}

struct NeuronLayer {
    neuron_paths: HashMap<Sensors, Vec<HiddenLayer>>
}



struct AnimalProperties {
    max_energy: f64,
    mass: f64,
    elasticity: f64,
    energy_digestion_rate: f64
}

impl AnimalProperties {
    fn create_offspring_properties(&self, other: &AnimalProperties) -> Self {
        todo!()
    }
}

struct Life {
    brain: Neuron,
    physical_properties: AnimalProperties,
}

impl Life {

}