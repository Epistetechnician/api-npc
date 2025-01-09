'use client'

import CabalTerminal from './agent'

export default function Home() {
  return (
    <main className="flex min-h-screen items-center justify-center bg-gradient-to-br from-black via-gray-900 to-slate-900 p-4">
      <div className="w-full max-w-[90vw] xl:max-w-[80vw]">
        <CabalTerminal />
      </div>
    </main>
  )
}