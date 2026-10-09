# Modified Windows helper

`ragent-srt-win.exe` is a modified build of Anthropic Sandbox Runtime's
Apache-2.0 `srt-win` helper, from tag `v0.0.78`, commit
`6f0ce155ccb136bda33a8a72201fe7f54fe47d9b`.
The upstream license is included at
`node_modules/@anthropic-ai/sandbox-runtime/LICENSE`.

RAgent changes only ACL update behavior: parent FILE_DELETE_CHILD denial is
object-only (not inherited over unrelated siblings); identical ACL updates are
skipped; object-only changes preserve inherited ACEs and protection state using
SetFileSecurityW. Working-tree grants and subtree read/write denies still use
SetNamedSecurityInfoW propagation. Account provisioning, WFP network filtering,
credential protection, sandbox process and job restrictions are unchanged.

Reviewable patch: `sandbox/patches/windows-acl.patch` in RAgent's source tree.
Rebuild: `scripts/build_sandbox_helper.py` with Rust and Windows build tools.
Upstream source: https://github.com/anthropics/sandbox-runtime
Artifact and patch checksums: `windows-helper.json`.
This is still an experimental Windows backend, not the Codex sandbox backend.

The distributed GNU-target build also includes compiler runtime notices in
`licenses/native/`: GCC's GPLv3 plus Runtime Library Exception, MinGW-w64
runtime notices and winpthreads license. It requires only Windows system DLLs;
the build toolchain itself is not distributed with the desktop EXE.
