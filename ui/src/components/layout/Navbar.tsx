import { NavLink } from 'react-router-dom'
import { Moon, Sun, LogOut, FlaskConical } from 'lucide-react'
import type { Session } from '@supabase/supabase-js'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { Avatar, AvatarFallback } from '@/components/ui/avatar'
import { useTheme } from '@/hooks/useTheme'
import { cn } from '@/lib/utils'

type NavbarProps = {
  health: 'loading' | 'online' | 'offline'
  authSession: Session | null
  onSignIn: () => void
  onSignOut: () => void
}

const NAV_LINKS = [
  { to: '/', label: 'Research' },
  { to: '/resources', label: 'Resources' },
  { to: '/agents', label: 'Agents' },
]

export function Navbar({ health, authSession, onSignIn, onSignOut }: NavbarProps) {
  const { theme, toggle } = useTheme()

  return (
    <header className="border-b bg-background sticky top-0 z-40">
      <div className="max-w-screen-xl mx-auto px-4 h-14 flex items-center gap-6">
        <div className="flex items-center gap-2 font-semibold text-sm">
          <FlaskConical size={18} className="text-primary" />
          Cortex
        </div>

        <nav className="flex items-center gap-1">
          {NAV_LINKS.map(({ to, label }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) =>
                cn(
                  'text-sm px-3 py-1.5 rounded-md transition-colors',
                  isActive
                    ? 'bg-muted text-foreground font-medium'
                    : 'text-muted-foreground hover:text-foreground hover:bg-muted/50',
                )
              }
            >
              {label}
            </NavLink>
          ))}
        </nav>

        <div className="flex-1" />

        <Badge variant="outline" className="text-xs gap-1.5">
          <span
            className={cn(
              'size-1.5 rounded-full',
              health === 'online'
                ? 'bg-green-500'
                : health === 'offline'
                  ? 'bg-red-500'
                  : 'bg-muted-foreground',
            )}
          />
          {health === 'online' ? 'Online' : health === 'offline' ? 'Offline' : 'Checking...'}
        </Badge>

        <Button variant="ghost" size="icon" onClick={toggle} aria-label="Toggle theme">
          {theme === 'dark' ? <Sun size={16} /> : <Moon size={16} />}
        </Button>

        {authSession ? (
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="icon" className="rounded-full">
                <Avatar className="size-8">
                  <AvatarFallback className="text-xs">
                    {authSession.user.email?.[0]?.toUpperCase() ?? '?'}
                  </AvatarFallback>
                </Avatar>
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuItem className="text-xs text-muted-foreground" disabled>
                {authSession.user.email}
              </DropdownMenuItem>
              <DropdownMenuItem onClick={onSignOut}>
                <LogOut size={14} className="mr-2" />
                Sign out
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        ) : (
          <Button size="sm" onClick={onSignIn}>
            Sign in with Google
          </Button>
        )}
      </div>
    </header>
  )
}
