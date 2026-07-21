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
  finishStreaming: (fullResponse: string, toolCalls?: ToolCallRecord[]) => void;
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

  finishStreaming: (fullResponse, toolCalls?: ToolCallRecord[]) => {
    const content = fullResponse || get().streamingContent;
    if (!content) {
      set({ isLoading: false, streamingContent: '', activeToolCall: null });
      return;
    }
    const msg: Message = {
      id: generateId(),
      role: 'assistant',
      content,
      toolCalls: toolCalls && toolCalls.length > 0 ? toolCalls : undefined,
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
