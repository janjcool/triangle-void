use vulkano::{
    instance::{
        Instance,
        InstanceExtensions,
        PhysicalDevice,
    },
    device::{
        Device,
        DeviceExtensions,
        Features,
    },
    buffer::{
        BufferUsage,
        CpuAccessibleBuffer,
    },
    command_buffer::{
        AutoCommandBufferBuilder,
        CommandBufferUsage,
        PrimaryCommandBuffer,
    },
    pipeline::{
        ComputePipeline,
        ComputePipelineAbstract,
    },
    descriptor::descriptor_set::PersistentDescriptorSet,
    sync::GpuFuture,
    Version,
};

use std::sync::Arc;

fn main() {
    let instance = Instance::new(None, Version::V1_1,
                                 &InstanceExtensions::none(), None).expect("failed to create instance");

    let physical = PhysicalDevice::enumerate(&instance).next().expect("no device available");

    for family in physical.queue_families() {
        println!("Found a queue family with {:?} queue(s)", family.queues_count());
    }

    let queue_family = physical.queue_families()
        .find(|&q| q.supports_graphics())
        .expect("couldn't find a graphical queue family");

    let (device, mut queues) = {
        Device::new(physical, &Features::none(), &DeviceExtensions::none(),
                    [(queue_family, 0.5)].iter().cloned()).expect("failed to create device")
    };
    let queue = queues.next().unwrap();

    let source_content = 0 .. 64;
    let source = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false,
                                                source_content).expect("failed to create buffer");

    let dest_content = (0 .. 64).map(|_| 0);
    let dest = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false,
                                              dest_content).expect("failed to create buffer");

    let mut builder = AutoCommandBufferBuilder::primary(device.clone(), queue.family(),
                                                        CommandBufferUsage::SimultaneousUse).unwrap(); // CommandBufferUsage::SimultaneousUse means that the command buffer can be
    // executed multiple times in parallel on different queues. If the command buffer breaks try to change the CommandBufferUsage
    builder.copy_buffer(source.clone(), dest.clone()).unwrap();
    let command_buffer = builder.build().unwrap();
    let finished = command_buffer.execute(queue.clone()).unwrap();

    finished.then_signal_fence_and_flush().unwrap()
        .wait(None).unwrap();
    let src_content = source.read().unwrap();
    let dest_content = dest.read().unwrap();
    assert_eq!(&*src_content, &*dest_content);

    let data_iter = 0 .. 65536;
    let data_buffer = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false,
                                                     data_iter).expect("failed to create buffer");

    mod cs {
        vulkano_shaders::shader!{
        ty: "compute",
        src: "
#version 450
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 0) buffer Data { uint data[]; } buf;
void main() {
    uint idx = gl_GlobalInvocationID.x;
    buf.data[idx] *= 12;
}"
        }
    }
    let shader = cs::Shader::load(device.clone())
        .expect("failed to create shader module");
    let compute_pipeline = Arc::new(ComputePipeline::new(device.clone(), &shader.main_entry_point(), &(), None)
        .expect("failed to create compute pipeline"));

    let layout = compute_pipeline.layout().descriptor_set_layout(0).unwrap();
    let set = Arc::new(PersistentDescriptorSet::start(layout.clone())
        .add_buffer(data_buffer.clone()).unwrap()
        .build().unwrap()
    );

    let mut builder = AutoCommandBufferBuilder::primary(device.clone(), queue.family(),
                                                        CommandBufferUsage::SimultaneousUse).unwrap();
    builder.dispatch([1024, 1, 1], compute_pipeline.clone(), set.clone(), (), []).unwrap();
    let command_buffer = builder.build().unwrap();

    let finished = command_buffer.execute(queue.clone()).unwrap();
    finished.then_signal_fence_and_flush().unwrap()
        .wait(None).unwrap();

    let content = data_buffer.read().unwrap();
    for (n, val) in content.iter().enumerate() {
        assert_eq!(*val, n as u32 * 12);
    }

    println!("Everything succeeded!");



}
