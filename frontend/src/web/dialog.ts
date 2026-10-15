// Stand-in for `@tauri-apps/plugin-dialog` used by the web build. Opens the
// browser's file/folder picker and registers the selected files.

import { register, registerFolder } from "./store";

type OpenOptions = {
  multiple?: boolean;
  directory?: boolean;
  filters?: { name?: string; extensions?: string[] }[];
};

export function open(opts: OpenOptions = {}): Promise<string | string[] | null> {
  return new Promise((resolve) => {
    const input = document.createElement("input");
    input.type = "file";
    if (opts.directory) {
      input.setAttribute("webkitdirectory", "");
      input.multiple = true;
    } else if (opts.multiple) {
      input.multiple = true;
      const exts = opts.filters?.flatMap((f) => f.extensions ?? []) ?? [];
      if (exts.length) input.accept = exts.map((e) => "." + e).join(",");
    }
    input.onchange = () => {
      const list = input.files;
      if (!list || list.length === 0) {
        resolve(null);
        return;
      }
      if (opts.directory) {
        resolve(registerFolder(list));
        return;
      }
      const keys = Array.from(list).map(register);
      resolve(opts.multiple ? keys : keys[0]);
    };
    input.click();
  });
}
