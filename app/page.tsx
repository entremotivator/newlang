"use client"
import { AgentCanvas } from "@/components/agent-canvas"
import { ComponentLibrary } from "@/components/component-library"
import { PropertiesPanel } from "@/components/properties-panel"
import { ExecutionPanel } from "@/components/execution-panel"
import { Header } from "@/components/header"
import { AgentProvider } from "@/contexts/agent-context"

export default function AgentBuilder() {
  return (
    <AgentProvider>
      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
        <Header />

        <div className="flex h-[calc(100vh-4rem)]">
          {/* Component Library Sidebar */}
          <div className="w-80 bg-white border-r border-slate-200 shadow-sm">
            <ComponentLibrary />
          </div>

          {/* Main Canvas Area */}
          <div className="flex-1 flex flex-col">
            <AgentCanvas />
          </div>

          {/* Properties & Execution Panel */}
          <div className="w-96 bg-white border-l border-slate-200 shadow-sm flex flex-col">
            <PropertiesPanel />
            <ExecutionPanel />
          </div>
        </div>
      </div>
    </AgentProvider>
  )
}
