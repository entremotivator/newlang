"use client"

import type React from "react"

import { createContext, useContext, useReducer, type ReactNode } from "react"

export interface NodeConfig {
  id: string
  type: NodeType
  name: string
  x: number
  y: number
  properties: Record<string, any>
  inputs: string[]
  outputs: string[]
}

export interface Connection {
  id: string
  fromNode: string
  toNode: string
  fromPort: string
  toPort: string
  condition?: string
}

export interface AgentFlow {
  nodes: NodeConfig[]
  connections: Connection[]
  metadata: Record<string, any>
}

export type NodeType =
  | "input"
  | "llm"
  | "tool"
  | "memory"
  | "router"
  | "output"
  | "prompt"
  | "parser"
  | "validator"
  | "webhook"

interface AgentState {
  flow: AgentFlow
  selectedNode: string | null
  isExecuting: boolean
  executionLog: string[]
  apiKey: string
  connectionMode: boolean
  connectionStart: string | null
}

type AgentAction =
  | { type: "ADD_NODE"; payload: NodeConfig }
  | { type: "UPDATE_NODE"; payload: { id: string; updates: Partial<NodeConfig> } }
  | { type: "DELETE_NODE"; payload: string }
  | { type: "ADD_CONNECTION"; payload: Connection }
  | { type: "DELETE_CONNECTION"; payload: string }
  | { type: "SELECT_NODE"; payload: string | null }
  | { type: "SET_API_KEY"; payload: string }
  | { type: "SET_EXECUTING"; payload: boolean }
  | { type: "ADD_LOG"; payload: string }
  | { type: "CLEAR_LOG" }
  | { type: "SET_CONNECTION_MODE"; payload: boolean }
  | { type: "SET_CONNECTION_START"; payload: string | null }
  | { type: "CLEAR_FLOW" }

const initialState: AgentState = {
  flow: { nodes: [], connections: [], metadata: {} },
  selectedNode: null,
  isExecuting: false,
  executionLog: [],
  apiKey: "",
  connectionMode: false,
  connectionStart: null,
}

function agentReducer(state: AgentState, action: AgentAction): AgentState {
  switch (action.type) {
    case "ADD_NODE":
      return {
        ...state,
        flow: {
          ...state.flow,
          nodes: [...state.flow.nodes, action.payload],
        },
      }

    case "UPDATE_NODE":
      return {
        ...state,
        flow: {
          ...state.flow,
          nodes: state.flow.nodes.map((node) =>
            node.id === action.payload.id ? { ...node, ...action.payload.updates } : node,
          ),
        },
      }

    case "DELETE_NODE":
      return {
        ...state,
        flow: {
          ...state.flow,
          nodes: state.flow.nodes.filter((node) => node.id !== action.payload),
          connections: state.flow.connections.filter(
            (conn) => conn.fromNode !== action.payload && conn.toNode !== action.payload,
          ),
        },
        selectedNode: state.selectedNode === action.payload ? null : state.selectedNode,
      }

    case "ADD_CONNECTION":
      return {
        ...state,
        flow: {
          ...state.flow,
          connections: [...state.flow.connections, action.payload],
        },
      }

    case "DELETE_CONNECTION":
      return {
        ...state,
        flow: {
          ...state.flow,
          connections: state.flow.connections.filter((conn) => conn.id !== action.payload),
        },
      }

    case "SELECT_NODE":
      return { ...state, selectedNode: action.payload }

    case "SET_API_KEY":
      return { ...state, apiKey: action.payload }

    case "SET_EXECUTING":
      return { ...state, isExecuting: action.payload }

    case "ADD_LOG":
      return {
        ...state,
        executionLog: [...state.executionLog, action.payload],
      }

    case "CLEAR_LOG":
      return { ...state, executionLog: [] }

    case "SET_CONNECTION_MODE":
      return { ...state, connectionMode: action.payload }

    case "SET_CONNECTION_START":
      return { ...state, connectionStart: action.payload }

    case "CLEAR_FLOW":
      return {
        ...state,
        flow: { nodes: [], connections: [], metadata: {} },
        selectedNode: null,
      }

    default:
      return state
  }
}

const AgentContext = createContext<{
  state: AgentState
  dispatch: React.Dispatch<AgentAction>
} | null>(null)

export function AgentProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(agentReducer, initialState)

  return <AgentContext.Provider value={{ state, dispatch }}>{children}</AgentContext.Provider>
}

export function useAgent() {
  const context = useContext(AgentContext)
  if (!context) {
    throw new Error("useAgent must be used within an AgentProvider")
  }
  return context
}
