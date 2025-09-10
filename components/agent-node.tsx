"use client"

import type React from "react"

import { useState, useRef } from "react"
import type { NodeConfig } from "@/contexts/agent-context"
import {
  MessageSquare,
  Bot,
  Wrench,
  Brain,
  GitBranch,
  Send,
  FileText,
  Search,
  CheckCircle,
  Webhook,
} from "lucide-react"

const nodeIcons = {
  input: MessageSquare,
  llm: Bot,
  tool: Wrench,
  memory: Brain,
  router: GitBranch,
  prompt: FileText,
  parser: Search,
  validator: CheckCircle,
  webhook: Webhook,
  output: Send,
}

const nodeColors = {
  input: "bg-green-500",
  llm: "bg-blue-500",
  tool: "bg-orange-500",
  memory: "bg-purple-500",
  router: "bg-red-500",
  prompt: "bg-indigo-500",
  parser: "bg-amber-500",
  validator: "bg-emerald-500",
  webhook: "bg-cyan-500",
  output: "bg-lime-500",
}

interface AgentNodeProps {
  node: NodeConfig
  isSelected: boolean
  isConnectionStart: boolean
  onClick: () => void
  onMove: (nodeId: string, x: number, y: number) => void
}

export function AgentNode({ node, isSelected, isConnectionStart, onClick, onMove }: AgentNodeProps) {
  const [isDragging, setIsDragging] = useState(false)
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 })
  const nodeRef = useRef<HTMLDivElement>(null)

  const IconComponent = nodeIcons[node.type]
  const colorClass = nodeColors[node.type]

  const handleMouseDown = (e: React.MouseEvent) => {
    if (e.button !== 0) return // Only left click

    const rect = nodeRef.current?.getBoundingClientRect()
    if (!rect) return

    setIsDragging(true)
    setDragOffset({
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
    })

    const handleMouseMove = (e: MouseEvent) => {
      if (!nodeRef.current?.parentElement) return

      const parentRect = nodeRef.current.parentElement.getBoundingClientRect()
      const newX = e.clientX - parentRect.left - dragOffset.x
      const newY = e.clientY - parentRect.top - dragOffset.y

      // Constrain to canvas bounds
      const constrainedX = Math.max(0, Math.min(newX, parentRect.width - 140))
      const constrainedY = Math.max(0, Math.min(newY, parentRect.height - 90))

      onMove(node.id, constrainedX, constrainedY)
    }

    const handleMouseUp = () => {
      setIsDragging(false)
      document.removeEventListener("mousemove", handleMouseMove)
      document.removeEventListener("mouseup", handleMouseUp)
    }

    document.addEventListener("mousemove", handleMouseMove)
    document.addEventListener("mouseup", handleMouseUp)
  }

  const handleClick = (e: React.MouseEvent) => {
    e.stopPropagation()
    if (!isDragging) {
      onClick()
    }
  }

  return (
    <div
      ref={nodeRef}
      className={`
        absolute w-35 h-22 cursor-pointer select-none transition-all duration-200
        ${isSelected ? "ring-2 ring-blue-500 ring-offset-2" : ""}
        ${isConnectionStart ? "ring-2 ring-orange-500 ring-offset-2" : ""}
        ${isDragging ? "z-50" : "z-10"}
      `}
      style={{
        left: node.x,
        top: node.y,
        transform: isDragging ? "scale(1.05)" : "scale(1)",
      }}
      onMouseDown={handleMouseDown}
      onClick={handleClick}
    >
      {/* Main Node Body */}
      <div
        className={`
        w-full h-full rounded-lg shadow-lg border-2 border-white
        ${colorClass} text-white
        hover:shadow-xl transition-shadow duration-200
        flex flex-col items-center justify-center p-3
      `}
      >
        {/* Icon */}
        <IconComponent className="w-6 h-6 mb-2" />

        {/* Name */}
        <div className="text-xs font-medium text-center leading-tight">{node.name}</div>

        {/* Status Indicator */}
        <div className="absolute -top-1 -right-1 w-3 h-3 bg-green-400 rounded-full border-2 border-white" />
      </div>

      {/* Input Port */}
      {node.inputs.length > 0 && (
        <div className="absolute left-0 top-1/2 -translate-y-1/2 -translate-x-1/2 w-3 h-3 bg-white border-2 border-slate-400 rounded-full" />
      )}

      {/* Output Port */}
      {node.outputs.length > 0 && (
        <div className="absolute right-0 top-1/2 -translate-y-1/2 translate-x-1/2 w-3 h-3 bg-white border-2 border-slate-400 rounded-full" />
      )}
    </div>
  )
}
