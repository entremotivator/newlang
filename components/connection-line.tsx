interface ConnectionLineProps {
  from: { x: number; y: number }
  to: { x: number; y: number }
}

export function ConnectionLine({ from, to }: ConnectionLineProps) {
  // Calculate control points for smooth bezier curve
  const dx = to.x - from.x
  const controlOffset = Math.abs(dx) * 0.5

  const pathData = `
    M ${from.x} ${from.y}
    C ${from.x + controlOffset} ${from.y}
      ${to.x - controlOffset} ${to.y}
      ${to.x} ${to.y}
  `

  // Calculate arrow position and angle
  const angle = Math.atan2(to.y - from.y, to.x - from.x)
  const arrowX = to.x - 15 * Math.cos(angle)
  const arrowY = to.y - 15 * Math.sin(angle)

  return (
    <g>
      {/* Connection Path */}
      <path d={pathData} fill="none" stroke="#64748b" strokeWidth="2" className="drop-shadow-sm" />

      {/* Arrow Head */}
      <polygon
        points="0,-4 8,0 0,4"
        fill="#64748b"
        transform={`translate(${arrowX}, ${arrowY}) rotate(${(angle * 180) / Math.PI})`}
      />
    </g>
  )
}
