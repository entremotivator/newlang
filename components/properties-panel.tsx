"use client"

import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Textarea } from "@/components/ui/textarea"
import { ScrollArea } from "@/components/ui/scroll-area"
import { useAgent } from "@/contexts/agent-context"
import { Trash2, Link } from "lucide-react"

export function PropertiesPanel() {
  const { state, dispatch } = useAgent()

  const selectedNode = state.selectedNode ? state.flow.nodes.find((n) => n.id === state.selectedNode) : null

  const handlePropertyChange = (key: string, value: any) => {
    if (!selectedNode) return

    dispatch({
      type: "UPDATE_NODE",
      payload: {
        id: selectedNode.id,
        updates: {
          properties: {
            ...selectedNode.properties,
            [key]: value,
          },
        },
      },
    })
  }

  const handleNameChange = (name: string) => {
    if (!selectedNode) return

    dispatch({
      type: "UPDATE_NODE",
      payload: {
        id: selectedNode.id,
        updates: { name },
      },
    })
  }

  const handleDeleteNode = () => {
    if (!selectedNode) return

    dispatch({ type: "DELETE_NODE", payload: selectedNode.id })
  }

  const toggleConnectionMode = () => {
    dispatch({ type: "SET_CONNECTION_MODE", payload: !state.connectionMode })
    dispatch({ type: "SET_CONNECTION_START", payload: null })
  }

  if (!selectedNode) {
    return (
      <div className="h-1/2 border-b border-slate-200">
        <div className="p-4 border-b border-slate-200">
          <h3 className="text-lg font-semibold text-slate-800">Properties</h3>
        </div>
        <div className="p-4 flex flex-col items-center justify-center h-32 text-slate-500">
          <p className="text-sm text-center">Select a node to edit its properties</p>
          <Button variant="outline" size="sm" className="mt-4 bg-transparent" onClick={toggleConnectionMode}>
            <Link className="w-4 h-4 mr-2" />
            {state.connectionMode ? "Exit Connect Mode" : "Connect Nodes"}
          </Button>
        </div>
      </div>
    )
  }

  return (
    <div className="h-1/2 border-b border-slate-200 flex flex-col">
      <div className="p-4 border-b border-slate-200">
        <h3 className="text-lg font-semibold text-slate-800">Properties</h3>
        <p className="text-sm text-slate-600 capitalize">{selectedNode.type} Node</p>
      </div>

      <ScrollArea className="flex-1 p-4">
        <div className="space-y-4">
          {/* Node Name */}
          <div>
            <Label htmlFor="node-name">Name</Label>
            <Input
              id="node-name"
              value={selectedNode.name}
              onChange={(e) => handleNameChange(e.target.value)}
              className="mt-1"
            />
          </div>

          {/* Dynamic Properties */}
          {Object.entries(selectedNode.properties).map(([key, value]) => (
            <div key={key}>
              <Label htmlFor={key} className="capitalize">
                {key.replace(/([A-Z])/g, " $1").toLowerCase()}
              </Label>

              {typeof value === "boolean" ? (
                <Select value={value.toString()} onValueChange={(val) => handlePropertyChange(key, val === "true")}>
                  <SelectTrigger className="mt-1">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="true">True</SelectItem>
                    <SelectItem value="false">False</SelectItem>
                  </SelectContent>
                </Select>
              ) : typeof value === "number" ? (
                <Input
                  id={key}
                  type="number"
                  value={value}
                  onChange={(e) => handlePropertyChange(key, Number.parseFloat(e.target.value))}
                  className="mt-1"
                />
              ) : typeof value === "string" && value.length > 50 ? (
                <Textarea
                  id={key}
                  value={value}
                  onChange={(e) => handlePropertyChange(key, e.target.value)}
                  className="mt-1"
                  rows={3}
                />
              ) : (
                <Input
                  id={key}
                  value={value}
                  onChange={(e) => handlePropertyChange(key, e.target.value)}
                  className="mt-1"
                />
              )}
            </div>
          ))}
        </div>
      </ScrollArea>

      <div className="p-4 border-t border-slate-200 space-y-2">
        <Button variant="outline" size="sm" className="w-full bg-transparent" onClick={toggleConnectionMode}>
          <Link className="w-4 h-4 mr-2" />
          {state.connectionMode ? "Exit Connect Mode" : "Connect Nodes"}
        </Button>

        <Button variant="destructive" size="sm" className="w-full" onClick={handleDeleteNode}>
          <Trash2 className="w-4 h-4 mr-2" />
          Delete Node
        </Button>
      </div>
    </div>
  )
}
