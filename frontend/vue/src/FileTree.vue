<script setup lang="ts">
import { computed, ref } from "vue";
const props = defineProps<{ files: string[]; directories: string[]; changed: Set<string>; prefix?: string; depth?: number }>();
const emit = defineEmits<{ open: [path: string]; folder: [path: string] }>();
const closed = ref(new Set<string>());
const treeIndent = computed(() => `${7 + (props.depth ?? 0) * 14}px`);
const nodes = computed(() => {
  const base = props.prefix ?? "";
  const children = new Map<string, { name: string; path: string; folder: boolean }>();
  for (const raw of [...props.directories, ...props.files]) {
    const path = raw.replace(/\\/g, "/");
    if (!path.startsWith(base)) continue;
    const remaining = path.slice(base.length), name = remaining.split("/")[0];
    if (!name) continue;
    const child = base + name;
    children.set(name, { name, path: child, folder: remaining.includes("/") || props.directories.includes(child) });
  }
  return [...children.values()].sort((a, b) => Number(b.folder) - Number(a.folder) || a.name.localeCompare(b.name));
});
function toggle(path: string) {
  const set = new Set(closed.value);
  if (set.has(path)) set.delete(path); else set.add(path);
  closed.value = set; emit("folder", path);
}
</script>
<template>
  <div class="vue-file-list">
    <div v-for="node in nodes" :key="node.path" :class="node.folder ? ['file-folder', { expanded: !closed.has(node.path) }] : undefined">
      <button v-if="node.folder" class="folder-entry" :style="{ '--tree-indent': treeIndent }" :title="node.path" @click="toggle(node.path)"><i class="tree-chevron">{{ closed.has(node.path) ? '›' : '⌄' }}</i><span><strong>{{ node.name }}</strong></span><em v-if="[...changed].some(p => p.startsWith(node.path + '/'))">●</em></button>
      <FileTree v-if="node.folder && !closed.has(node.path)" class="folder-children" :prefix="node.path + '/'" :files="files" :directories="directories" :changed="changed" :depth="(depth ?? 0) + 1" @open="emit('open', $event)" @folder="emit('folder', $event)" />
      <button v-else-if="!node.folder" class="file-entry" :class="{ changed: changed.has(node.path) }" :style="{ '--tree-indent': treeIndent }" :title="node.path" @click="emit('open', node.path)"><i class="tree-spacer" /><span><strong>{{ node.name }}</strong></span><em v-if="changed.has(node.path)">已修改</em></button>
    </div>
  </div>
</template>
