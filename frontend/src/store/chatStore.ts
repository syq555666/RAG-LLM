import { create } from 'zustand';
import type { Message, ToolCallRecord } from '../types/chat';
import { generateId } from '../utils/format';

interface ChatState {
  messages: Message[];
  isLoading: boolean;
  streamingContent: string;
  activeToolCall: ToolCallRecord | null;
  suggestions: string[];
  sessionId: string | null;

  setSessionId: (id: string) => void;
  addUserMessage: (content: string) => Message;
  addAssistantMessage: (content: string) => void;
  startStreaming: () => void;
  appendToken: (content: string) => void;
  setToolCall: (tc: ToolCallRecord | null) => void;
  finishStreaming: (fullResponse: string) => void;
  setSuggestions: (suggestions: string[]) => void;
  loadHistory: (msgs: { role: string; content: string }[]) => void;
  clearMessages: () => void;
}

export const useChatStore = create<ChatState>((set, get) => ({
  messages: [],
  isLoading: false,
  streamingContent: '',
  activeToolCall: null,
  suggestions: [],
  sessionId: null,

  setSessionId: (id) => set({ sessionId: id }),

  addUserMessage: (content) => {
    const msg: Message = { id: generateId(), role: 'user', content };
    set((s) => ({ messages: [...s.messages, msg], suggestions: [] }));
    return msg;
  },

  addAssistantMessage: (content) => {
    const msg: Message = { id: generateId(), role: 'assistant', content };
    set((s) => ({ messages: [...s.messages, msg] }));
  },

  startStreaming: () =>
    set({ isLoading: true, streamingContent: '', activeToolCall: null }),

  appendToken: (content) =>
    set((s) => ({ streamingContent: s.streamingContent + content })),

  setToolCall: (tc) => set({ activeToolCall: tc }),

  finishStreaming: (fullResponse) => {
    const msg: Message = {
      id: generateId(),
      role: 'assistant',
      content: fullResponse || get().streamingContent,
      toolCalls: get().activeToolCall ? [get().activeToolCall!] : undefined,
    };
    set({
      messages: [...get().messages, msg],
      isLoading: false,
      streamingContent: '',
      activeToolCall: null,
    });
  },

  setSuggestions: (suggestions) => set({ suggestions }),

  loadHistory: (msgs) => {
    const messages: Message[] = msgs.map((m) => ({
      id: generateId(),
      role: m.role as 'user' | 'assistant',
      content: m.content,
    }));
    set({ messages });
  },

  clearMessages: () =>
    set({ messages: [], streamingContent: '', activeToolCall: null, suggestions: [] }),
}));
