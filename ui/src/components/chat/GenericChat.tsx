import { ChatThreadContainer } from '@/components/chat/ChatThreadContainer'
import { workspaceChatTransport } from '@/components/chat/transports'

type Props = {
  accessToken: string
  activeSessionId: string | null
  onSessionActivated: (id: string | null) => void
  onSessionsChanged: () => void
}

export function GenericChat({ accessToken, activeSessionId, onSessionActivated, onSessionsChanged }: Props) {
  return (
    <ChatThreadContainer
      transport={workspaceChatTransport}
      accessToken={accessToken}
      activeSessionId={activeSessionId}
      onSessionActivated={onSessionActivated}
      onSessionsChanged={onSessionsChanged}
      title="Chat"
      subtitle="Workspace-wide chat across your uploaded resources."
      emptyState="Ask anything about your resources or the web."
    />
  )
}
