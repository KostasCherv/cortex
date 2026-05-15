import { useMemo } from 'react'
import { ChatThreadContainer } from '@/components/chat/ChatThreadContainer'
import { createAgentChatTransport } from '@/components/chat/transports'
import type { RagAgent } from '@/types'

type Props = {
  agent: RagAgent
  accessToken: string
  activeSessionId: string | null
  onSessionActivated: (id: string | null) => void
  onSessionsChanged: () => void
}

export function AgentChat({ agent, accessToken, activeSessionId, onSessionActivated, onSessionsChanged }: Props) {
  const transport = useMemo(() => createAgentChatTransport(agent.agent_id), [agent.agent_id])

  return (
    <ChatThreadContainer
      transport={transport}
      accessToken={accessToken}
      activeSessionId={activeSessionId}
      onSessionActivated={onSessionActivated}
      onSessionsChanged={onSessionsChanged}
      title={agent.name}
      subtitle={agent.description || undefined}
      emptyState={`Ask ${agent.name} about its linked resources.`}
      resourceLabel={`${agent.linked_resource_ids.length} resources`}
      defaultWebSearchEnabled={false}
    />
  )
}
