# Upstream replacement API contracts

Review the target vLLM revision and its actual callers before editing an override. For
every patched attribute or exposed registration, map the local file and symbol to the
exact upstream callable, schema or interface using `mapping-put`. Cover retained and
conditionally installed replacements, plus the effective binding after a replacement is
retired. Separate bindings need separate contract IDs, including when they select the
same local callable. Mark an infrastructure-only file with a reason in
`no_api_replacements`. Use the upstream extension contract for a platform-specific
registration that implements an upstream interface.

## Record the target surface

Include `api_contracts` in the task's `checks-put` document. The source gate requires
coverage for Python files under `vllm_metax/patch/` and `vllm_metax/registry/`, and for
plugin Python files in tasks classified as API, dispatch or lifecycle changes. Mapped
replacement symbols need their own contract even if the file's inventory has a reason.

```json
{
  "changes": ["none"],
  "reason": "Retained replacement reviewed against the current target",
  "api_contracts": [
    {
      "id": "encode",
      "local": "vllm_metax/patch/example.py",
      "local_symbol": "encode",
      "binding": {
        "entry": "vllm.example.Encoder.encode",
        "callable": "vllm_metax.patch.example.encode"
      },
      "surface": "python",
      "mode": "exact",
      "target": {"signature": "(items, /, *, options=None)"},
      "cases": ["valid_call", "invalid_call", "explicit_options", "omitted_options"]
    }
  ]
}
```

Capture the target API at the same binding level as the effective API, such as a bound
method, constructor or factory. Record the exact callable selected by the lookup or
patch assignment. For `surface: custom_op`, use `schema` and `fake_signature` in
`target`. Use `fake_signature: null` for a target without a fake implementation. Compare
the visible schema including types, defaults, mutation, aliases and
return structure. When a wrapper exposes metadata from another callable, exercise its
real binding and forwarding as well.

Use `mode: adapter` for a necessary platform deviation. Record `reason` and
`removal_condition` in that contract and include `adapter_forwarding` among its cases.
Keep the adapter behind the target interface when possible. If a Python signature must
differ, review and exercise all supported caller forms through the effective binding,
including positional-only/keyword-only parameters, defaults, explicit overrides and
validation behavior as applicable. For an internal kernel or provider difference,
translate arguments and preserve results, state and error behavior at the public boundary.
CustomOp schemas and fake signatures stay exact in both modes.

## Exercise the activated entry point

In fresh processes, capture the original binding from the selected upstream checkout
and the effective binding after MetaX activation. The component runner starts with
`VLLM_PLUGINS` empty. Enable the normal MetaX activation path in the candidate process
before resolving the replacement. For a patch, call through the actual patched attribute
or affected consumer. For a registration, perform its normal lookup and
call the result, including real and fake paths where the operator has both. Cover the
selected configuration, meaningful omitted and explicit values, and affected aliases,
fallbacks, unrelated keys and repeat activation where reachable. Observe return values,
mutations, state, and exceptions. Use representative invalid calls to prove rejection
semantics. Run model-free component checks during source adaptation and validate complete
model paths at the normal final gate.

The `contract` check writes an `api_contracts` object keyed by contract ID into
`METAX_UPGRADE_ORIGINS_FILE`. The tool freezes declarations in the check's
`acceptance.json`. For the preceding example, the child reports the measured binding,
upstream and activated surfaces and each actual case result.

```json
{
  "api_contracts": {
    "encode": {
      "binding": {
        "entry": "vllm.example.Encoder.encode",
        "callable": "vllm_metax.patch.example.encode"
      },
      "upstream": {"signature": "(items, /, *, options=None)"},
      "active": {"signature": "(items, /, *, options=None)"},
      "cases": {
        "valid_call": true,
        "invalid_call": true,
        "explicit_options": true,
        "omitted_options": true
      }
    }
  }
}
```

Save the probe and supporting call inputs with the check evidence. Add any normal
matrix configuration fields required by the task. A passing source phase requires
the declared binding and caller cases to be current. Re-run them when the target or
effective binding changes, including after final review of an early blocking patch.
