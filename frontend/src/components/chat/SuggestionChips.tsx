import React from 'react';

interface SuggestionChipsProps {
  suggestions: string[];
  onSelect: (suggestion: string) => void;
}

export const SuggestionChips: React.FC<SuggestionChipsProps> = ({ suggestions, onSelect }) => {
  if (suggestions.length === 0) return null;

  return (
    <div className="suggestion-chips">
      {suggestions.map((s, i) => (
        <button key={i} className="suggestion-chip" onClick={() => onSelect(s)}>
          {s}
        </button>
      ))}
    </div>
  );
};
