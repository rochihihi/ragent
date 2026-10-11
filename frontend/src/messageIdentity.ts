/** A transport retry reuses its identity; a successful intentional send does not. */
export class MessageIdentity {
  private pending = new Map<string, { content: string; id: string }>();
  constructor(private newId: () => string = () => crypto.randomUUID()) {}
  acquire(session: string, content: string): string {
    const prior = this.pending.get(session);
    if (prior?.content === content) return prior.id;
    const id = this.newId();
    this.pending.set(session, { content, id });
    return id;
  }
  acknowledge(session: string, id: string): void {
    if (this.pending.get(session)?.id === id) this.pending.delete(session);
  }
}
