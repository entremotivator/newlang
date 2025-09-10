"use client"

import type React from "react"

import { useRef, useCallback } from "react"
import { useAgent } from "@/contexts/agent-context"
import { AgentNode } from "./agent-node"
import { ConnectionLine } from "./connection-line"

export function AgentCanvas() {
  const { state, dispatch } = useAgent()
  const canvasRef = useRef<HTMLDivElement>(null)

  const handleCanvasClick = useCallback(
    (e: React.MouseEvent) => {
      if (e.target === canvasRef.current) {
        dispatch({ type: "SELECT_NODE", payload: null })
        if (state.connectionMode) {
          dispatch({ type: "SET_CONNECTION_MODE", payload: false })
          dispatch({ type: "SET_CONNECTION_START", payload: null })
        }
      }
    },
    [dispatch, state.connectionMode],
  )

  const handleNodeClick = useCallback(
    (nodeId: string) => {
      if (state.connectionMode) {
        if (!state.connectionStart) {
          dispatch({ type: "SET_CONNECTION_START", payload: nodeId })
        } else if (state.connectionStart !== nodeId) {
          // Create connection
          const connectionId = `conn_${Date.now()}`
          dispatch({
            type: "ADD_CONNECTION",
            payload: {
              id: connectionId,
              fromNode: state.connectionStart,
              toNode: nodeId,
              fromPort: "output",
              toPort: "input",
            },
          })
          dispatch({ type: "SET_CONNECTION_MODE", payload: false })
          dispatch({ type: "SET_CONNECTION_START", payload: null })
        }
      } else {
        dispatch({ type: "SELECT_NODE", payload: nodeId })
      }
    },
    [dispatch, state.connectionMode, state.connectionStart],
  )

  const handleNodeMove = useCallback(
    (nodeId: string, x: number, y: number) => {
      dispatch({
        type: "UPDATE_NODE",
        payload: {
          id: nodeId,
          updates: { x, y },
        },
      })
    },
    [dispatch],
  )

  return (
    <div className="flex-1 relative overflow-hidden bg-slate-50">
      {/* Grid Background */}
      <div
        className="absolute inset-0 opacity-20"
        style={{
          backgroundImage: `
            linear-gradient(to right, #e2e8f0 1px, transparent 1px),
            linear-gradient(to bottom, #e2e8f0 1px, transparent 1px)
          `,
          backgroundSize: "20px 20px",
        }}
      />

      {/* Canvas */}
      <div ref={canvasRef} className="relative w-full h-full cursor-default" onClick={handleCanvasClick}>
        {/* Connection Lines */}
        <svg className="absolute inset-0 w-full h-full pointer-events-none">
          {state.flow.connections.map((connection) => {
            const fromNode = state.flow.nodes.find((n) => n.id === connection.fromNode)
            const toNode = state.flow.nodes.find((n) => n.id === connection.toNode)

            if (!fromNode || !toNode) return null

            return (
              <ConnectionLine
                key={connection.id}
                from={{ x: fromNode.x + 140, y: fromNode.y + 45 }}
                to={{ x: toNode.x, y: toNode.y + 45 }}
              />
            )
          })}
        </svg>

        {/* Nodes */}
        {state.flow.nodes.map((node) => (
          <AgentNode
            key={node.id}
            node={node}
            isSelected={state.selectedNode === node.id}
            isConnectionStart={state.connectionStart === node.id}
            onClick={() => handleNodeClick(node.id)}
            onMove={handleNodeMove}
          />
        ))}

        {/* Connection Mode Indicator */}
        {state.connectionMode && (
          <div className="absolute top-4 left-4 bg-blue-500 text-white px-3 py-2 rounded-lg shadow-lg">
            {state.connectionStart ? "Select target node" : "Select source node"}
          </div>
        )}
      </div>
    </div>
  )
}
