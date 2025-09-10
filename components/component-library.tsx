"use client"

import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import { useAgent, type NodeType } from "@/contexts/agent-context"
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

const componentConfigs = {
  input: {
    icon: MessageSquare,
    color: "bg-green-500",
    description: "User input capture",
    properties: {
      inputType: "text",
      placeholder: "Enter your message...",
      required: true,
      validation: "",
    },
  },
  llm: {
    icon: Bot,
    color: "bg-blue-500",
    description: "Language model processing",
    properties: {
      model: "gpt-4o",
      temperature: 0.7,
      maxTokens: 2000,
      systemPrompt: "You are a helpful assistant.",
      stream: true,
    },
  },
  tool: {
    icon: Wrench,
    color: "bg-orange-500",
    description: "External tools and APIs",
    properties: {
      toolType: "web_search",
      endpoint: "",
      method: "GET",
      timeout: 30,
    },
  },
  memory: {
    icon: Brain,
    color: "bg-purple-500",
    description: "Conversation memory",
    properties: {
      memoryType: "conversation_buffer",
      maxTokens: 1000,
      returnMessages: true,
    },
  },
  router: {
    icon: GitBranch,
    color: "bg-red-500",
    description: "Decision routing logic",
    properties: {
      routingType: "semantic",
      conditions: {},
      defaultRoute: "",
      confidenceThreshold: 0.8,
    },
  },
  prompt: {
    icon: FileText,
    color: "bg-indigo-500",
    description: "Prompt template engine",
    properties: {
      template: "Answer the question: {question}",
      variables: ["question"],
      templateType: "simple",
    },
  },
  parser: {
    icon: Search,
    color: "bg-amber-500",
    description: "Output parsing and formatting",
    properties: {
      parserType: "json",
      schema: {},
      formatTemplate: "",
      errorHandling: "strict",
    },
  },
  validator: {
    icon: CheckCircle,
    color: "bg-emerald-500",
    description: "Input/output validation",
    properties: {
      validationRules: {},
      errorMessage: "Validation failed",
      strictMode: true,
    },
  },
  webhook: {
    icon: Webhook,
    color: "bg-cyan-500",
    description: "External webhook integration",
    properties: {
      url: "",
      method: "POST",
      headers: {},
      authType: "none",
    },
  },
  output: {
    icon: Send,
    color: "bg-lime-500",
    description: "Final output formatting",
    properties: {
      format: "text",
      template: "",
      postProcess: false,
    },
  },
}

export function ComponentLibrary() {
  const { dispatch } = useAgent()

  const addNode = (type: NodeType) => {
    const config = componentConfigs[type]
    const nodeId = `${type}_${Date.now()}`

    dispatch({
      type: "ADD_NODE",
      payload: {
        id: nodeId,
        type,
        name: `${type.charAt(0).toUpperCase() + type.slice(1)} Node`,
        x: Math.random() * 400 + 100,
        y: Math.random() * 300 + 100,
        properties: config.properties,
        inputs: type === "input" ? [] : ["input"],
        outputs: type === "output" ? [] : ["output"],
      },
    })
  }

  return (
    <div className="h-full flex flex-col">
      <div className="p-4 border-b border-slate-200">
        <h2 className="text-lg font-semibold text-slate-800">Component Library</h2>
        <p className="text-sm text-slate-600 mt-1">Drag components to build your agent flow</p>
      </div>

      <ScrollArea className="flex-1 p-4">
        <div className="space-y-3">
          {Object.entries(componentConfigs).map(([type, config]) => {
            const IconComponent = config.icon
            return (
              <Button
                key={type}
                variant="outline"
                className="w-full h-auto p-4 flex flex-col items-start gap-2 hover:shadow-md transition-shadow bg-transparent"
                onClick={() => addNode(type as NodeType)}
              >
                <div className="flex items-center gap-3 w-full">
                  <div className={`p-2 rounded-lg ${config.color} text-white`}>
                    <IconComponent className="w-5 h-5" />
                  </div>
                  <div className="flex-1 text-left">
                    <div className="font-medium capitalize">{type}</div>
                    <div className="text-xs text-slate-500 mt-1">{config.description}</div>
                  </div>
                </div>
              </Button>
            )
          })}
        </div>
      </ScrollArea>
    </div>
  )
}
