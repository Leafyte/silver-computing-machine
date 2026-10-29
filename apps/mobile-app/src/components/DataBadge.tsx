import React from 'react';
import { View, Text, StyleSheet } from 'react-native';

interface DataBadgeProps {
  label: string;
  tone?: 'default' | 'success' | 'warning' | 'preview' | 'risk';
}

export const DataBadge: React.FC<DataBadgeProps> = ({ label, tone = 'preview' }) => {
  const getBadgeStyle = () => {
    switch (tone) {
      case 'success':
        return {
          bg: '#E2F4EE',
          border: '#A8DBC9',
          text: '#205A46'
        };
      case 'warning':
        return {
          bg: '#FDF4EB',
          border: '#FBD6B0',
          text: '#B26B00'
        };
      case 'risk':
        return {
          bg: '#FBECEB',
          border: '#F5C2C0',
          text: '#C62828'
        };
      case 'preview':
      default:
        return {
          bg: '#F3F4F6',
          border: '#E5E7EB',
          text: '#4A5565'
        };
    }
  };

  const colors = getBadgeStyle();

  return (
    <View style={[styles.badge, { backgroundColor: colors.bg, borderColor: colors.border }]}>
      <Text style={[styles.text, { color: colors.text }]}>{label}</Text>
    </View>
  );
};

const styles = StyleSheet.create({
  badge: {
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 999,
    borderWidth: 1,
    alignSelf: 'flex-start',
  },
  text: {
    fontSize: 10,
    fontWeight: '700',
    letterSpacing: 0.2,
  }
});
