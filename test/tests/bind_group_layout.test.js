import { test, group } from "../test.js";
import { WgslReflect } from "../../wgsl_reflect.module.js";

export async function run() {
  await group("bind_group_layout", async function () {
    const shader = `
  struct Uniforms {
      model: mat4x4<f32>,   // offset(0)  align(16) size(64)
      color: vec4<f32>,     // offset(64) align(16) size(16)
  }                         //            align(16) size(80)

  struct Small {
      a: vec2<f32>,         // offset(0)  align(8)  size(8)
  }                         //            align(8)  size(8)

  struct Particle {
      pos: vec4<f32>,
      vel: vec4<f32>,
  }

  struct Particles {
      count: atomic<u32>,   // offset(0)  align(4)  size(4)
      data: array<Particle>,// offset(16) align(16) runtime-sized
  }

  @group(0) @binding(0) var<uniform> uniforms: Uniforms;
  @group(0) @binding(1) var<storage, read> particlesIn: Particles;
  @group(0) @binding(2) var<storage, read_write> particlesOut: Particles;
  @group(1) @binding(0) var tex: texture_2d<f32>;
  @group(1) @binding(1) var samp: sampler;
  @group(1) @binding(2) var depthTex: texture_depth_2d;
  @group(1) @binding(3) var msaaTex: texture_multisampled_2d<f32>;
  @group(1) @binding(4) var cmpSamp: sampler_comparison;
  @group(2) @binding(0) var<uniform> unusedUniforms: Uniforms;
  @group(2) @binding(1) var<uniform> small: Small;

  @vertex
  fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
      let p = particlesIn.data[i].pos;
      return uniforms.model * p;
  }

  @fragment
  fn fs() -> @location(0) vec4<f32> {
      let c = textureSample(tex, samp, vec2<f32>(0.0));
      let d = textureSampleCompare(depthTex, cmpSamp, vec2<f32>(0.0), 0.5);
      let m = textureLoad(msaaTex, vec2<i32>(0), 0);
      return uniforms.color * c + vec4<f32>(d) + m;
  }

  @compute @workgroup_size(1)
  fn cs() {
      particlesOut.data[0].vel = particlesIn.data[0].vel;
  }`;

    let reflect = null;
    let layouts = null;

    await test("reflect", function (test) {
      reflect = new WgslReflect(shader);
      layouts = reflect.getBindGroupLayouts();
      test.true(layouts !== null);
    });

    await test("groups are dense and sorted", function (test) {
      test.equals(layouts.length, 3);
      test.equals(layouts[0].group, 0);
      test.equals(layouts[1].group, 1);
      test.equals(layouts[2].group, 2);
    });

    await test("uniform buffer entry", function (test) {
      const e = layouts[0].entries[0];
      test.equals(e.binding, 0);
      test.equals(e.buffer.type, "uniform");
      test.equals(e.buffer.hasDynamicOffset, false);
      test.equals(e.buffer.minBindingSize, 80);
      test.equals(e.visibility, ["vertex", "fragment"]);
    });

    await test("storage buffer entries", function (test) {
      const readOnly = layouts[0].entries[1];
      test.equals(readOnly.binding, 1);
      test.equals(readOnly.buffer.type, "read-only-storage");
      // Runtime-sized trailing array: size stops at the array's offset.
      test.equals(readOnly.buffer.minBindingSize, 16);
      test.equals(readOnly.visibility, ["vertex", "compute"]);

      const readWrite = layouts[0].entries[2];
      test.equals(readWrite.binding, 2);
      test.equals(readWrite.buffer.type, "storage");
      test.equals(readWrite.buffer.minBindingSize, 16);
      test.equals(readWrite.visibility, ["compute"]);
    });

    await test("texture entries", function (test) {
      const tex = layouts[1].entries[0];
      test.equals(tex.binding, 0);
      test.equals(tex.texture.sampleType, "float");
      test.equals(tex.texture.viewDimension, "2d");
      test.equals(tex.texture.multisampled, false);
      test.equals(tex.visibility, ["fragment"]);

      const depth = layouts[1].entries[2];
      test.equals(depth.binding, 2);
      test.equals(depth.texture.sampleType, "depth");
      test.equals(depth.texture.viewDimension, "2d");
      test.equals(depth.texture.multisampled, false);

      const msaa = layouts[1].entries[3];
      test.equals(msaa.binding, 3);
      test.equals(msaa.texture.sampleType, "float");
      test.equals(msaa.texture.viewDimension, "2d");
      test.equals(msaa.texture.multisampled, true);
    });

    await test("sampler entries", function (test) {
      const samp = layouts[1].entries[1];
      test.equals(samp.binding, 1);
      test.equals(samp.sampler.type, "filtering");

      const cmp = layouts[1].entries[4];
      test.equals(cmp.binding, 4);
      test.equals(cmp.sampler.type, "comparison");
    });

    await test("unreferenced resource gets empty visibility", function (test) {
      const unused = layouts[2].entries[0];
      test.equals(unused.binding, 0);
      test.equals(unused.buffer.type, "uniform");
      test.equals(unused.visibility, []);
    });

    await test("uniform minBindingSize is 16-byte aligned", function (test) {
      const small = layouts[2].entries[1];
      test.equals(small.binding, 1);
      test.equals(small.buffer.type, "uniform");
      // Struct size is 8, rounded up to 16 for the uniform address space.
      test.equals(small.buffer.minBindingSize, 16);
    });

    await test("entries sorted by binding", function (test) {
      const outOfOrder = new WgslReflect(`
        @group(0) @binding(2) var<uniform> c: Uniforms;
        @group(0) @binding(0) var<uniform> a: Uniforms;
        @group(0) @binding(1) var<uniform> b: Uniforms;
        struct Uniforms { x: f32, };`);
      const entries = outOfOrder.getBindGroupLayouts()[0].entries;
      test.equals(entries.length, 3);
      test.equals(entries[0].binding, 0);
      test.equals(entries[1].binding, 1);
      test.equals(entries[2].binding, 2);
    });

    await test("non-zero group stays dense", function (test) {
      const r = new WgslReflect(`
        @group(3) @binding(0) var<uniform> u: Uniforms;
        struct Uniforms { x: f32, };`);
      const l = r.getBindGroupLayouts();
      test.equals(l.length, 1);
      test.equals(l[0].group, 3);
      test.equals(l[0].entries.length, 1);
    });

    await test("dynamicBuffers sets hasDynamicOffset", function (test) {
      const l = reflect.getBindGroupLayouts({
        dynamicBuffers: [{ group: 0, binding: 0 }],
      });
      test.equals(l[0].entries[0].buffer.hasDynamicOffset, true);
      test.equals(l[0].entries[1].buffer.hasDynamicOffset, false);
    });

    await test("dynamicBuffers on storage buffer throws", function (test) {
      let error = null;
      try {
        reflect.getBindGroupLayouts({
          dynamicBuffers: [{ group: 0, binding: 1 }],
        });
      } catch (e) {
        error = e;
      }
      test.true(error !== null, "expected an error");
      test.true(error.message.indexOf("group 0") != -1, error?.message);
      test.true(error.message.indexOf("binding 1") != -1, error?.message);
    });

    await test("dynamicBuffers on missing binding throws", function (test) {
      let error = null;
      try {
        reflect.getBindGroupLayouts({
          dynamicBuffers: [{ group: 5, binding: 9 }],
        });
      } catch (e) {
        error = e;
      }
      test.true(error !== null, "expected an error");
      test.true(error.message.indexOf("group 5") != -1, error?.message);
      test.true(error.message.indexOf("binding 9") != -1, error?.message);
    });
  });
}
