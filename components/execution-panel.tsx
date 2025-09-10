"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Badge } from "@/components/ui/badge"
import { useAgent } from "@/contexts/agent-context"
import { Play, Square, Trash2 } from "lucide-react"

export function ExecutionPanel() {
  const { state, dispatch } = useAgent()
  const [testInput, setTestInput] = useState("What is the capital of France?")

  const handleExecute = async () => {
    if (!state.apiKey) {
      dispatch({ type: "ADD_LOG", payload: "Error: OpenAI API key not provided" })
      return
    }

    if (state.flow.nodes.length === 0) {
      dispatch({ type: "ADD_LOG", payload: "Error: No nodes in the flow" })
      return
    }

    dispatch({ type: "SET_EXECUTING", payload: true })
    dispatch({ type: "ADD_LOG", payload: `Starting execution with input: "${testInput}"` })

    try {
      // Simulate agent execution
      await new Promise((resolve) => setTimeout(resolve, 2000))
      dispatch({ type: "ADD_LOG", payload: "Execution completed successfully" })
      dispatch({ type: "ADD_LOG", payload: `Output: This is a simulated response to "${testInput}"` })
    } catch (error) {
      dispatch({ type: "ADD_LOG", payload: `Error: ${error}` })
    } finally {
      dispatch({ type: "SET_EXECUTING", payload: false })
    }
  }

  const handleStop = () => {
    dispatch({ type: "SET_EXECUTING", payload: false })
    dispatch({ type: "ADD_LOG", payload: "Execution stopped by user" })
  }

  const handleClearLog = () => {
    dispatch({ type: "CLEAR_LOG" })
  }

  return (
    <div className="h-1/2 flex flex-col">
      <div className="p-4 border-b border-slate-200">
        <h3 className="text-lg font-semibold text-slate-800">Execution</h3>
        <div className="flex items-center gap-2 mt-2">
          <Badge variant={state.apiKey ? "default" : "secondary"}>
            {state.apiKey ? "API Ready" : "API Not Connected"}
          </Badge>
          <Badge variant="outline">{state.flow.nodes.length} nodes</Badge>
        </div>
      </div>

      <div className="p-4 space-y-4">
        {/* Test Input */}
        <div>
          <label className="text-sm font-medium text-slate-700 mb-2 block">Test Input</label>
          <Textarea
            value={testInput}
            onChange={(e) => setTestInput(e.target.value)}
            placeholder="Enter test input for your agent..."
            rows={3}
            className="resize-none"
          />
        </div>

        {/* Control Buttons */}
        <div className="flex gap-2">
          <Button
            onClick={handleExecute}
            disabled={state.isExecuting || !state.apiKey || state.flow.nodes.length === 0}
            className="flex-1"
          >
            <Play className="w-4 h-4 mr-2" />
            {state.isExecuting ? "Running..." : "Run"}
          </Button>

          <Button variant="outline" onClick={handleStop} disabled={!state.isExecuting}>
            <Square className="w-4 h-4" />
          </Button>

          <Button variant="outline" onClick={handleClearLog} size="icon">
            <Trash2 className="w-4 h-4" />
          </Button>
        </div>
      </div>

      {/* Execution Log */}
      <div className="flex-1 border-t border-slate-200">
        <div className="p-4 pb-2">
          <h4 className="text-sm font-medium text-slate-700">Execution Log</h4>
        </div>

        <ScrollArea className="flex-1 px-4 pb-4">
          <div className="space-y-2">
            {state.executionLog.length === 0 ? (
              <p className="text-sm text-slate-500 italic">No execution logs yet. Run your agent to see output here.</p>
            ) : (
              state.executionLog.map((log, index) => (
                <div
                  key={index}
                  className={`text-sm p-2 rounded font-mono ${
                    log.startsWith("Error:")
                      ? "bg-red-50 text-red-700 border border-red-200"
                      : log.startsWith("Output:")
                        ? "bg-green-50 text-green-700 border border-green-200"
                        : "bg-slate-50 text-slate-700 border border-slate-200"
                  }`}
                >
                  {log}
                </div>
              ))
            )}
          </div>
        </ScrollArea>
      </div>
    </div>
  )
}
