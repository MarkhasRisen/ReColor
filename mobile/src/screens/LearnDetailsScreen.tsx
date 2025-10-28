import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  SafeAreaView,
  ScrollView,
} from 'react-native';

const topicContent: { [key: string]: { title: string; content: string[] } } = {
  basics: {
    title: 'What is Color Blindness?',
    content: [
      'Color blindness, or color vision deficiency (CVD), is a condition where a person has difficulty distinguishing certain colors.',
      'It\'s typically inherited and affects the cone cells in the retina, which are responsible for color perception.',
      'Most people with CVD can see colors, but certain shades appear similar or indistinguishable.',
      'It\'s not a form of blindness - people with CVD have full visual acuity in most cases.',
    ],
  },
  types: {
    title: 'Types of CVD',
    content: [
      '🔴 Protanomaly/Protanopia (Red Deficiency)\n• Affects red cones\n• Red appears darker\n• Affects ~1% of males\n• Less common than deutan',
      '🟢 Deuteranomaly/Deuteranopia (Green Deficiency)\n• Affects green cones\n• Most common type\n• Affects ~5% of males\n• Green appears less vibrant',
      '🔵 Tritanomaly/Tritanopia (Blue Deficiency)\n• Affects blue cones\n• Very rare\n• Affects <1% of population\n• Blue and yellow confusion',
      '⚫ Achromatopsia (Complete Color Blindness)\n• Extremely rare\n• See only in grayscale\n• Often includes light sensitivity',
    ],
  },
  statistics: {
    title: 'How Common is CVD?',
    content: [
      '📊 Global Statistics:\n• ~8% of males have red-green CVD\n• ~0.5% of females have red-green CVD\n• ~300 million people worldwide',
      '🔬 Why More Males?\nCVD is X-linked genetic trait. Males have one X chromosome, so one affected gene = CVD. Females need two affected genes.',
      '🌍 Ethnic Variations:\n• Northern European descent: ~8-9%\n• Asian populations: ~4-5%\n• African descent: ~2-4%',
    ],
  },
  living: {
    title: 'Living with CVD',
    content: [
      '🚦 Traffic Signals:\n• Remember position (top=red, bottom=green)\n• Use brightness cues\n• Modern LED signals are clearer',
      '👔 Clothing & Fashion:\n• Label clothes by color\n• Use color identification apps\n• Organize by color families\n• Ask friends/family for input',
      '🎮 Digital Screens:\n• Use colorblind-friendly modes\n• Adjust contrast and brightness\n• Enable color filters (iOS/Android)\n• Use pattern/shape indicators',
      '💼 Career Considerations:\n• Most careers are accessible\n• Some restrictions: pilots, electricians\n• Many accommodations available\n• Technology helps bridge gaps',
    ],
  },
  genetics: {
    title: 'Genetics & Inheritance',
    content: [
      '🧬 Inheritance Pattern:\nRed-green CVD is X-linked recessive. Males need one affected X, females need two.',
      '👨‍👩‍👧 Family Patterns:\n• Father with CVD + Mother carrier = 50% sons affected\n• Father normal + Mother carrier = 50% sons affected\n• Daughters of affected fathers are carriers',
      '🔬 Genetic Cause:\nMutations in OPN1LW (red) or OPN1MW (green) genes on X chromosome affect cone photopigments.',
    ],
  },
  technology: {
    title: 'Technology & Solutions',
    content: [
      '📱 Mobile Apps:\n• ReColor (this app!)\n• Color identification tools\n• Filter and enhancement apps\n• AR color overlay tools',
      '👓 Special Glasses:\n• EnChroma glasses\n• Pilestone glasses\n• Enhance color discrimination\n• Don\'t "cure" but help some',
      '💻 Software Solutions:\n• OS-level color filters\n• Browser extensions\n• Accessibility features\n• Design tools with CVD modes',
      '🎨 Design Considerations:\n• Use patterns and shapes\n• Add text labels\n• Sufficient contrast\n• Test with CVD simulations',
    ],
  },
};

const LearnDetailsScreen = ({ route }: any) => {
  const { topic } = route.params || { topic: 'basics' };
  const content = topicContent[topic] || topicContent.basics;

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView contentContainerStyle={styles.content}>
        <Text style={styles.title}>{content.title}</Text>
        
        {content.content.map((paragraph, index) => (
          <View key={index} style={styles.paragraphContainer}>
            <Text style={styles.paragraph}>{paragraph}</Text>
          </View>
        ))}

        <View style={styles.footer}>
          <Text style={styles.footerText}>
            💡 Remember: This information is educational. Always consult an eye care professional for diagnosis and advice.
          </Text>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F5F5F5',
  },
  content: {
    padding: 20,
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 20,
  },
  paragraphContainer: {
    backgroundColor: '#FFF',
    borderRadius: 12,
    padding: 18,
    marginBottom: 15,
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
  },
  paragraph: {
    fontSize: 16,
    color: '#333',
    lineHeight: 26,
  },
  footer: {
    backgroundColor: '#E8F4F8',
    borderRadius: 12,
    padding: 18,
    marginTop: 20,
    borderWidth: 1,
    borderColor: '#B4D7E5',
  },
  footerText: {
    fontSize: 14,
    color: '#2C5F7A',
    lineHeight: 22,
    textAlign: 'center',
  },
});

export default LearnDetailsScreen;
