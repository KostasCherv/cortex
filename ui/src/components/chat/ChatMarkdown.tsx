import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { cn } from '@/lib/utils'

export const assistantAvatarClassName =
  'flex size-7 shrink-0 items-center justify-center rounded-full bg-muted text-[10px] font-bold text-foreground'

export const assistantBubbleClassName =
  'rounded-2xl rounded-bl-sm bg-muted px-3 py-2 text-sm text-foreground'

const proseClassName = cn(
  'prose prose-sm max-w-none text-foreground dark:prose-invert',
  'prose-headings:text-foreground prose-p:text-foreground prose-strong:text-foreground',
  'prose-li:text-foreground prose-ol:text-foreground prose-ul:text-foreground',
  'prose-a:text-primary prose-a:no-underline hover:prose-a:underline',
  'prose-p:my-0 prose-ul:my-1 prose-ol:my-1 prose-li:my-0 prose-pre:my-2',
  'prose-code:before:content-none prose-code:after:content-none',
  'prose-table:my-2 prose-th:border prose-th:border-border prose-th:px-2 prose-th:py-1 prose-th:text-left',
  'prose-td:border prose-td:border-border prose-td:px-2 prose-td:py-1',
)

type Props = {
  content: string
  className?: string
}

export function ChatMarkdown({ content, className }: Props) {
  return (
    <div className={cn('overflow-x-auto', className)}>
      <div className={proseClassName}>
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
      </div>
    </div>
  )
}
