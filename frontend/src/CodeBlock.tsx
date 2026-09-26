import { Children, ComponentProps, isValidElement, useRef, useState } from "react";

// Copy from rendered textContent: highlighted spans must not alter source text.
export function CodeBlock({ children }: ComponentProps<"pre">) {
  const source = useRef<HTMLPreElement>(null);
  const [status, setStatus] = useState("复制");
  const child = Children.toArray(children).find(isValidElement);
  const language = child && isValidElement<{ className?: string }>(child)
    ? /(?:^|\s)language-([^\s]+)/.exec(child.props.className ?? "")?.[1]
    : undefined;
  async function copy() {
    try {
      await navigator.clipboard.writeText(source.current?.textContent ?? "");
      setStatus("已复制");
    } catch {
      setStatus("复制失败，请手动选择");
    }
  }
  return <section className="code-block">
    <header><span>{language || "代码"}</span><button type="button" onClick={copy} onBlur={() => setStatus("复制")}>{status}</button></header>
    <pre ref={source}>{children}</pre>
  </section>;
}
