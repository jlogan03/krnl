//! rspirv 0.13 does not load OpExecutionModeId into the global modes section.
use rspirv::{
    binary::{Consumer, ParseAction},
    dr::{Instruction, Module, ModuleHeader},
    spirv::Op,
};

#[derive(Default)]
struct Loader {
    inner: rspirv::dr::Loader,
    id_modes: Vec<Instruction>,
}

impl Consumer for Loader {
    fn initialize(&mut self) -> ParseAction {
        self.inner.initialize()
    }
    fn finalize(&mut self) -> ParseAction {
        self.inner.finalize()
    }
    fn consume_header(&mut self, header: ModuleHeader) -> ParseAction {
        self.inner.consume_header(header)
    }
    fn consume_instruction(&mut self, inst: Instruction) -> ParseAction {
        if inst.class.opcode == Op::ExecutionModeId {
            self.id_modes.push(inst);
            ParseAction::Continue
        } else {
            self.inner.consume_instruction(inst)
        }
    }
}

pub(crate) fn load_words(words: impl AsRef<[u32]>) -> anyhow::Result<Module> {
    let mut loader = Loader::default();
    rspirv::binary::parse_words(words.as_ref(), &mut loader)
        .map_err(|err| anyhow::anyhow!(err.to_string()))?;
    let mut module = loader.inner.module();
    module.execution_modes.extend(loader.id_modes);
    Ok(module)
}

/// Remove globals and helper functions belonging only to other entry points.
/// SPIRV-Tools conservatively skips some DCE passes with FloatControls2.
#[allow(dead_code)] // Shared with krnlc; the runtime only needs the loader.
pub(crate) fn remove_unused_globals(module: &mut Module) {
    use rspirv::{
        dr::Operand,
        spirv::{BuiltIn, Decoration},
    };
    use std::collections::{HashMap, HashSet};
    // Remove unused generated thread-descriptor construction and its input
    // loads. These operations have no observable effects (volatile loads are
    // excluded); this does not rewrite or reorder floating-point arithmetic.
    loop {
        let referenced: HashSet<_> = module
            .functions
            .iter()
            .flat_map(|f| f.all_inst_iter())
            .flat_map(|inst| inst.operands.iter().filter_map(|op| op.id_ref_any()))
            .collect();
        let mut changed = false;
        for block in module.functions.iter_mut().flat_map(|f| &mut f.blocks) {
            block.instructions.retain(|inst| {
                let pure = matches!(
                    inst.class.opcode,
                    Op::CompositeConstruct
                        | Op::CompositeExtract
                        | Op::CopyObject
                        | Op::IMul
                        | Op::IAdd
                        | Op::ISub
                ) || (inst.class.opcode == Op::Load && inst.operands.len() == 1);
                let dead = pure && inst.result_id.is_some_and(|id| !referenced.contains(&id));
                changed |= dead;
                !dead
            });
        }
        if !changed {
            break;
        }
    }
    let references = |inst: &Instruction| -> Vec<u32> {
        inst.result_type
            .into_iter()
            .chain(inst.operands.iter().filter_map(|op| op.id_ref_any()))
            .collect()
    };
    let mut edges: HashMap<u32, Vec<u32>> = module
        .types_global_values
        .iter()
        .map(|inst| (inst.result_id.unwrap(), references(inst)))
        .collect();
    for function in &module.functions {
        edges.insert(
            function.def.as_ref().unwrap().result_id.unwrap(),
            function
                .all_inst_iter()
                .flat_map(|inst| inst.result_id.into_iter().chain(references(inst)))
                .collect(),
        );
    }
    let mut pending: Vec<_> = module
        .entry_points
        .iter()
        .map(|inst| inst.operands[1].unwrap_id_ref())
        .chain(module.execution_modes.iter().flat_map(references))
        .collect();
    for inst in &module.annotations {
        if inst.operands.get(1) == Some(&Operand::Decoration(Decoration::BuiltIn))
            && inst.operands.get(2) == Some(&Operand::BuiltIn(BuiltIn::WorkgroupSize))
        {
            pending.push(inst.operands[0].unwrap_id_ref());
        }
    }
    let mut used = HashSet::new();
    while let Some(id) = pending.pop() {
        if used.insert(id) {
            if let Some(deps) = edges.get(&id) {
                pending.extend(deps);
            }
            for inst in &module.annotations {
                if inst.operands.first().and_then(|op| op.id_ref_any()) == Some(id) {
                    pending.extend(references(inst));
                }
            }
        }
    }
    module
        .functions
        .retain(|function| used.contains(&function.def.as_ref().unwrap().result_id.unwrap()));
    module
        .types_global_values
        .retain(|inst| used.contains(&inst.result_id.unwrap()));
    module.annotations.retain(|inst| {
        inst.operands
            .first()
            .and_then(|op| op.id_ref_any())
            .is_some_and(|id| used.contains(&id))
    });
    module.debug_names.retain(|inst| {
        inst.operands
            .first()
            .and_then(|op| op.id_ref_any())
            .is_some_and(|id| used.contains(&id))
    });
    for entry in &mut module.entry_points {
        let interfaces = entry.operands.split_off(3);
        entry.operands.extend(
            interfaces
                .into_iter()
                .filter(|op| used.contains(&op.unwrap_id_ref())),
        );
    }
}
