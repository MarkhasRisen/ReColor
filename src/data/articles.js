export const ARTICLES = [
  {
    id: 'a1',
    title: 'Designing for Accessibility',
    category: 'Design',
    readTime: '7 min read',
    coverImage: require('../../assets/art.png'),
    summary: 'Inclusive design benefits everyone. A few simple principles make a big difference.',
    sections: [
      { type: 'heading', text: "Do's & Don'ts" },
      {
        type: 'dos_donts',
        dos: ['Add text labels to charts', 'Use patterns + color', 'Ensure 4.5:1 contrast'],
        donts: ['Use color alone for info', 'Red text on green bg', 'Skip alt text on icons'],
      },
      { type: 'heading', text: 'Contrast Ratio Guide' },
      {
        type: 'contrast',
        items: [
          { ratio: '4.5:1', label: 'Normal text', pass: true },
          { ratio: '3:1', label: 'Large text (18px+)', pass: true },
          { ratio: '1.5:1', label: 'Decorative', pass: false },
        ],
      },
      { type: 'heading', text: 'Colorblind-Safe Palettes' },
      {
        type: 'palettes',
        items: [
          { colors: ['#0077BB', '#EE7733'], name: 'Blue & Orange', safe: 'Red-green safe' },
          { colors: ['#009988', '#EE3377'], name: 'Teal & Purple', safe: 'Multi-CVD safe' },
          { colors: ['#222222', '#DDAA33'], name: 'Black & Yellow', safe: 'High contrast' },
        ],
      },
      { type: 'heading', text: 'Testing Tools' },
      { type: 'tools', items: ['Coblis', 'Color Oracle', 'WebAIM Checker', 'Figma Plugin'] },
    ],
  },
  {
    id: 'a2',
    title: 'Types of Color Blindness',
    category: 'Guide',
    readTime: '6 min read',
    coverImage: require('../../assets/eye.png'),
    summary: 'CVD ranges from mild color shifts to complete inability to perceive color.',
    sections: [
      { type: 'heading', text: 'Severity Spectrum' },
      { type: 'severity_spectrum' },
      { type: 'heading', text: 'CVD Types' },
      {
        type: 'cvd_types',
        items: [
          {
            name: 'Protanomaly',
            sub: 'Red-weak vision',
            colors: ['#C0392B', '#922B21', '#641E16', '#4A0E0E'],
            desc: 'Red appears darker; red-green distinctions are difficult.',
          },
          {
            name: 'Deuteranomaly',
            sub: 'Green-weak vision',
            colors: ['#27AE60', '#1E8449', '#196F3D', '#145A32'],
            desc: 'Most common CVD — affects ~5% of males worldwide.',
          },
          {
            name: 'Tritanomaly',
            sub: 'Blue-weak vision',
            colors: ['#2980B9', '#1A5276', '#154360', '#0E2F44'],
            desc: 'Blue-yellow distinctions are difficult. Rarest type.',
          },
          {
            name: 'Monochromacy',
            sub: 'Complete color blindness',
            colors: ['#BDC3C7', '#95A5A6', '#7F8C8D', '#566573'],
            desc: 'Sees only shades of gray. Rarest and most severe form.',
          },
        ],
      },
    ],
  },
  {
    id: 'a3',
    title: 'What is Color Vision Deficiency?',
    category: 'Science',
    readTime: '4 min read',
    coverImage: require('../../assets/access.png'),
    summary: 'A condition where the ability to distinguish certain colors is reduced — far more common than you think.',
    sections: [
      {
        type: 'stats',
        items: [
          { value: '1 in 12', label: 'men affected' },
          { value: '1 in 200', label: 'women affected' },
          { value: '300M+', label: 'worldwide' },
        ],
      },
      { type: 'heading', text: 'How the Eye Sees Color' },
      {
        type: 'paragraph',
        text: 'Your retina has 3 cone types. When one is missing or weak, CVD occurs.',
      },
      {
        type: 'cone_diagram',
        cones: [
          { label: 'L-Cone', sub: 'Red ~700nm', color: '#E74C3C' },
          { label: 'M-Cone', sub: 'Green ~540nm', color: '#2ECC71' },
          { label: 'S-Cone', sub: 'Blue ~420nm', color: '#3498DB' },
        ],
      },
      { type: 'heading', text: 'Common Causes' },
      {
        type: 'causes',
        items: ['Genetic (X-linked)', 'Aging', 'Eye disease', 'Medication'],
      },
    ],
  },
  {
    id: 'a4',
    title: 'Tips for Daily Life',
    category: 'Daily Life',
    readTime: '5 min read',
    coverImage: require('../../assets/domore.png'),
    summary: 'With the right strategies and tools, you can navigate a color-coded world with confidence.',
    sections: [
      {
        type: 'bento_grid',
        items: [
          { icon: 'phone-portrait-outline', title: 'Tech Tools', desc: 'Use color-identifier apps & accessibility filters on your phone' },
          { icon: 'shirt-outline', title: 'Wardrobe', desc: 'Label clothes by texture tags or use a color-ID app while shopping' },
          { icon: 'business-outline', title: 'Workplace', desc: 'Ask for pattern-based charts; inform HR about your CVD needs' },
          { icon: 'navigate-outline', title: 'Daily Life', desc: 'Traffic lights: red on top, green on bottom — use position, not color' },
        ],
      },
      { type: 'heading', text: 'Quick Wins' },
      {
        type: 'quick_wins',
        items: [
          'Enable color filters in Accessibility settings',
          'Cook by temperature, not meat color',
          'Ask for charts with patterns + color',
          'Use high-contrast mode for readability',
        ],
      },
    ],
  },
];
