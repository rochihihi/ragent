<script setup lang="ts">
import { computed } from "vue";
import MarkdownIt from "markdown-it";
const props = defineProps<{ content: string }>();
// Never allow HTML from model/tool output to execute in the workspace.
const markdown = new MarkdownIt({ html: false, linkify: true, breaks: true });
const rendered = computed(() => markdown.render(props.content));
const sections = computed(() => props.content.split(/\n{2,}/).slice(1).map(block => {
  const [title, ...lines] = block.split("\n");
  return { title, lines: lines.map(line => line.replace(/^\s*[-*]\s+/, "")) };
}));
</script>
<template>
  <div v-if="content.startsWith('任务完成\n\n')" class="completion-result">
    <header><span>✓</span><div><strong>任务已完成</strong><small>结果与执行状态已保存；验证以实际工具结果为准</small></div></header>
    <div class="completion-grid"><section v-for="(part, index) in sections" :key="index"><h4>{{ part.title }}</h4><div v-for="(line, i) in part.lines" :key="i" class="completion-item"><i>·</i><span>{{ line }}</span></div></section></div>
  </div>
  <div v-else class="message-content" v-html="rendered" />
</template>
