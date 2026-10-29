import React, { useState } from 'react';
import { ArrowRight, LockKeyhole } from 'lucide-react';
import { apiClient } from '../../api/client';

interface LoginViewProps {
  onAuthenticated: () => void;
}

export const LoginView: React.FC<LoginViewProps> = ({ onAuthenticated }) => {
  const [username, setUsername] = useState('dispatcher');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [submitting, setSubmitting] = useState(false);

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    setSubmitting(true);
    setError('');
    try {
      await apiClient.login(username.trim(), password);
      onAuthenticated();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Could not sign in. Check the API connection and credentials.');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <main className="min-h-screen bg-[#F5F3ED] flex items-center justify-center px-5 py-10 text-[#141F1A]">
      <section className="w-full max-w-md rounded-[28px] bg-white border border-[#EAE6DF] shadow-xl p-8">
        <div className="w-12 h-12 rounded-2xl bg-[#E8F5EE] text-[#1E5A44] flex items-center justify-center mb-6"><LockKeyhole className="w-5 h-5" /></div>
        <p className="text-[10px] font-bold uppercase tracking-[0.2em] text-[#1E5A44]">arQ Dispatch OS</p>
        <h1 className="mt-2 text-2xl font-extrabold tracking-tight">Dispatcher sign in</h1>
        <p className="mt-2 text-sm leading-6 text-[#68756F]">Sign in to the demo workspace. Phone driver activity will appear in the operational event stream.</p>
        <form onSubmit={submit} className="mt-7 space-y-4">
          <label className="block text-xs font-bold">Username
            <input autoComplete="username" value={username} onChange={(event) => setUsername(event.target.value)} className="mt-1.5 w-full rounded-xl border border-[#E5E0D8] px-3.5 py-3 text-sm outline-none focus:border-[#1E5A44]" />
          </label>
          <label className="block text-xs font-bold">Password
            <input type="password" autoComplete="current-password" value={password} onChange={(event) => setPassword(event.target.value)} className="mt-1.5 w-full rounded-xl border border-[#E5E0D8] px-3.5 py-3 text-sm outline-none focus:border-[#1E5A44]" />
          </label>
          {error && <p role="alert" className="rounded-xl bg-[#FBECEB] px-3 py-2 text-xs text-[#C6534C]">{error}</p>}
          <button disabled={submitting || !username.trim() || !password} className="w-full rounded-xl bg-[#141F1A] px-4 py-3 text-sm font-bold text-white flex items-center justify-center gap-2 disabled:opacity-50">
            {submitting ? 'Signing in…' : 'Sign in'} {!submitting && <ArrowRight className="w-4 h-4" />}
          </button>
        </form>
        <div className="mt-6 rounded-2xl border border-[#E5E0D8] bg-[#F8F7F3] p-4 text-xs">
          <p className="font-bold">Demo credentials</p>
          <p className="mt-1 text-[#68756F]">dispatcher / <span className="font-mono text-[#141F1A]">dispatch-demo-2026</span></p>
          <p className="mt-3 text-[10px] leading-5 text-[#7D8D85]">Local demo account only. Configure demo users on the backend before exposing it beyond localhost.</p>
        </div>
      </section>
    </main>
  );
};
