/* tslint:disable */
/* eslint-disable */

export class WasmInference {
  private constructor();
  free(): void;
  [Symbol.dispose](): void;
  nextChunk(): Promise<string | undefined>;
  clearStream(): void;
  startStream(prompt: string, max_tokens: number | null | undefined, temperature: number, top_k?: number | null, context_window?: number | null): string;
}

export function loadModel(model_bytes: Uint8Array, vocab_json: string, config_json?: string | null, start_viz?: boolean | null): Promise<WasmInference>;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly __wbg_wasminference_free: (a: number, b: number) => void;
  readonly loadModel: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => number;
  readonly wasminference_clearStream: (a: number) => void;
  readonly wasminference_nextChunk: (a: number) => number;
  readonly wasminference_startStream: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => void;
  readonly __wasm_bindgen_func_elem_12731: (a: number, b: number, c: number) => void;
  readonly __wasm_bindgen_func_elem_12728: (a: number, b: number) => void;
  readonly __wasm_bindgen_func_elem_2965: (a: number, b: number, c: number) => void;
  readonly __wasm_bindgen_func_elem_1602: (a: number, b: number) => void;
  readonly __wasm_bindgen_func_elem_12741: (a: number, b: number) => void;
  readonly __wasm_bindgen_func_elem_107016: (a: number, b: number, c: number) => void;
  readonly __wasm_bindgen_func_elem_107014: (a: number, b: number) => void;
  readonly __wasm_bindgen_func_elem_12733: (a: number, b: number, c: number, d: number) => void;
  readonly __wasm_bindgen_func_elem_107070: (a: number, b: number, c: number, d: number) => void;
  readonly __wbindgen_export: (a: number, b: number) => number;
  readonly __wbindgen_export2: (a: number, b: number, c: number, d: number) => number;
  readonly __wbindgen_export3: (a: number) => void;
  readonly __wbindgen_export4: (a: number, b: number, c: number) => void;
  readonly __wbindgen_add_to_stack_pointer: (a: number) => number;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
* Instantiates the given `module`, which can either be bytes or
* a precompiled `WebAssembly.Module`.
*
* @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
*
* @returns {InitOutput}
*/
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
* If `module_or_path` is {RequestInfo} or {URL}, makes a request and
* for everything else, calls `WebAssembly.instantiate` directly.
*
* @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
*
* @returns {Promise<InitOutput>}
*/
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
