"use client"

import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { useAgent } from "@/contexts/agent-context"
import { Bot, Save, FolderOpen, Trash2, Key } from "lucide-react"

export function Header() {
  const { state, dispatch } = useAgent()

  const handleClearFlow = () => {
    dispatch({ type: "CLEAR_FLOW" })
  }

  const handleApiKeyChange = (value: string) => {
    dispatch({ type: "SET_API_KEY", payload: value })
  }

  return (
    <header className="h-16 bg-white border-b border-slate-200 shadow-sm">
      <div className="flex items-center justify-between h-full px-6">
        <div className="flex items-center gap-3">
          <Bot className="w-8 h-8 text-blue-600" />
          <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
            LangChain Agent Builder
          </h1>
        </div>

        <div className="flex items-center gap-4">
          {/* API Key Input */}
          <div className="flex items-center gap-2">
            <Key className="w-4 h-4 text-slate-500" />
            <Input
              type="password"
              placeholder="OpenAI API Key"
              value={state.apiKey}
              onChange={(e) => handleApiKeyChange(e.target.value)}
              className="w-48"
            />
            <Badge variant={state.apiKey ? "default" : "secondary"}>
              {state.apiKey ? "Connected" : "Not Connected"}
            </Badge>
          </div>

          {/* Action Buttons */}
          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm">
              <FolderOpen className="w-4 h-4 mr-2" />
              Load
            </Button>
            <Button variant="outline" size="sm">
              <Save className="w-4 h-4 mr-2" />
              Save
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={handleClearFlow}
              className="text-red-600 hover:text-red-700 bg-transparent"
            >
              <Trash2 className="w-4 h-4 mr-2" />
              Clear
            </Button>
          </div>

          {/* Node Count */}
          <Badge variant="outline">{state.flow.nodes.length} nodes</Badge>
        </div>
      </div>
    </header>
  )
}
