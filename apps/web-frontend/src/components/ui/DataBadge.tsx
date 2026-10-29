import React from 'react';
import { ProvenanceBadge } from '../../types/api';

interface DataBadgeProps {
  badge: ProvenanceBadge;
  className?: string;
  size?: 'sm' | 'md';
}

export const DataBadge: React.FC<DataBadgeProps> = ({ badge, className = '', size = 'md' }) => {
  // Styles based on provenance tier
  const getBadgeStyle = (b: ProvenanceBadge) => {
    switch (b) {
      case 'Validated':
        return 'bg-[#E2F4EE] text-[#205A46] border-[#A8DBC9]';
      case 'Validator failed':
        return 'bg-[#FBECEB] text-[#C6534C] border-[#F2B8B5]';
      case 'Exact (enumeration)':
      case 'Exact (financial cover)':
        return 'bg-[#EBF1EC] text-[#17231F] border-[#D9E6DC]';
      case 'QpiAI Opt · local solver':
        return 'bg-[#EFF4FF] text-[#2F5BEA] border-[#C7D7FE]';
      case 'QAOA (warm start)':
      case 'QCloud':
      case 'VQC · local statevector':
        return 'bg-[#F5F0FF] text-[#8E3BD1] border-[#E0D0FB]';
      case 'GBDT + Platt (classical)':
        return 'bg-[#FDF4EB] text-[#B26B00] border-[#FBD6B0]';
      case 'Classical re-plan':
        return 'bg-[#E2F4EE] text-[#205A46] border-[#A8DBC9]';
      case 'Fallback':
        return 'bg-[#FFF7ED] text-[#C2410C] border-[#FFEDD5]';
      case 'Simulated · held-out run':
        return 'bg-[#F3F4F6] text-[#4B5563] border-[#E5E7EB]';
      case 'Preview':
      default:
        return 'bg-[#F8F9FA] text-[#68756F] border-[#E5E7EB]';
    }
  };

  const sizeClasses = size === 'sm' ? 'px-2 py-0.5 text-[11px]' : 'px-2.5 py-1 text-xs';

  return (
    <span
      className={`inline-flex items-center font-medium border rounded-full whitespace-nowrap tabular-nums tracking-tight ${sizeClasses} ${getBadgeStyle(
        badge
      )} ${className}`}
      title={`Data Provenance: ${badge}`}
    >
      <span className="w-1.5 h-1.5 rounded-full mr-1.5 bg-current opacity-70" />
      {badge}
    </span>
  );
};
