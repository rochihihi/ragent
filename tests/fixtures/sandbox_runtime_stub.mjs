// UNIT TEST ONLY: no OS isolation. Redirect the library, preserve the production
// bridge's argv/trampoline. Never loaded by RAgent's sanitized host broker.
import { registerHooks } from "node:module";
const fake = `import { appendFileSync } from 'node:fs';
const log=(event,data)=>{if(process.env.RAGENT_TEST_ACL_LOG) appendFileSync(process.env.RAGENT_TEST_ACL_LOG,JSON.stringify({event,...data})+'\\n')};
export const installWindowsSandboxAsync=async()=>{throw new Error('not allowed in unit test')};
export const checkWindowsDependenciesAsync=async()=>({errors:process.env.RAGENT_TEST_DEPENDENCY_FAILURE?['missing account']:[],warnings:[]});
export const resolveSrtWin=()=>({});export const VENDORED_SRT_WIN_EXE='unit-test-only';
export const getWindowsSandboxUserStatusAsync=async()=>({provisioned:true,credPresent:true,sid:'S-1-5-21-test'});
export const grantWindowsAcl=options=>{log('grant',options);if(process.env.RAGENT_TEST_GRANT_FAILURE) throw new Error('partial grant failure')};
export const revokeWindowsAcl=options=>{log('revoke',options);return process.env.RAGENT_TEST_CLEANUP_FAILURE?undefined:[]};
export const restoreWindowsAcl=options=>{log('restore',options);if(process.env.RAGENT_TEST_RESTORE_THROW) throw new Error('restore helper failure');return process.env.RAGENT_TEST_RESTORE_FAILURE?[{status:'failed'}]:[]};
export const SandboxManager={ checkDependenciesAsync:async()=>({errors:[],warnings:[]}),
initialize:async()=>{log('initialize',{});if(process.env.RAGENT_TEST_INIT_FAILURE) throw new Error('WFP verification failure')},
isSandboxingEnabled:()=>true,reset:async()=>{log('reset',{})},
wrapWithSandboxArgv:async(command,shell)=>({argv:[shell.exe,...shell.args,command],env:process.env}),
wrapWithSandbox:async(command)=>command };`;
registerHooks({ resolve(specifier, context, nextResolve) {
  if (specifier === "@anthropic-ai/sandbox-runtime") return {
    url: "data:text/javascript," + encodeURIComponent(fake), shortCircuit: true,
  };
  return nextResolve(specifier, context);
}});
