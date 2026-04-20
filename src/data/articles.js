export const ARTICLES = [
  {
    id: 'a1',
    title: 'Designing for Accessibility',
    category: 'Design',
    readTime: '7 min read',
    coverImage: require('../../assets/art.png'),
    summary: 'Inclusive design benefits everyone. A few simple principles make a big difference.',
    sections: [
      {
        type: 'heading',
        text: "Do's & Don'ts",
      },
      {
        type: 'dos_donts',
        dos: [
          'Add text labels to charts',
          'Use patterns + color',
          'Ensure 4.5:1 contrast',
        ],
        donts: [
          'Use color alone for info',
          'Red text on green bg',
          'Skip alt text on icons',
        ],
      },
      {
        type: 'heading',
        text: 'Contrast Ratio Guide',
      },
      {
        type: 'contrast',
        items: [
          { ratio: '4.5:1', label: 'Normal text', pass: true },
          { ratio: '3:1', label: 'Large text (18px+)', pass: true },
          { ratio: '1.5:1', label: 'Decorative', pass: false },
        ],
      },
      {
        type: 'heading',
        text: 'Colorblind-Safe Palettes',
      },
      {
        type: 'palettes',
        items: [
          { colors: ['#0077BB', '#EE7733'], name: 'Blue & Orange', safe: 'Red-green safe' },
          { colors: ['#009988', '#EE3377'], name: 'Teal & Purple', safe: 'Multi-CVD safe' },
          { colors: ['#222222', '#DDAA33'], name: 'Black & Yellow', safe: 'High contrast' },
        ],
      },
      {
        type: 'heading',
        text: 'Testing Tools',
      },
      {
        type: 'tools',
        items: ['Coblis', 'Color Oracle', 'WebAIM Checker', 'Figma Plugin'],
      },
    ],
  },
  {
    id: 'a2',
    title: 'Understanding Color Vision Deficiency',
    category: 'Education',
    readTime: '5 min read',
    coverImage: require('../../assets/eye.png'),
    summary: 'Learn the science behind how colour blindness affects approximately 8% of males and 0.5% of females worldwide.',
    sections: [
      {
        type: 'paragraph',
        text: 'Colour vision deficiency (CVD) is caused by missing or malfunctioning cone cells in the retina. There are three types of cone cells, each sensitive to different wavelengths of light: red (long), green (medium), and blue (short).',
      },
      {
        type: 'heading',
        text: 'Types of CVD',
      },
      {
        type: 'list',
        items: [
          { title: 'Protanopia', desc: 'Absent red cones. Red appears dark, confused with black or dark brown.' },
          { title: 'Deuteranopia', desc: 'Absent green cones. Most common form of red-green blindness.' },
          { title: 'Tritanopia', desc: 'Absent blue cones. Rare. Blue and green appear similar.' },
          { title: 'Achromatopsia', desc: 'Complete colour blindness. Sees only shades of grey.' },
        ],
      },
      {
        type: 'heading',
        text: 'Prevalence',
      },
      {
        type: 'paragraph',
        text: 'CVD affects approximately 300 million people worldwide. It is more common in males (8%) than females (0.5%) due to the X-linked inheritance pattern of most forms of colour vision deficiency.',
      },
    ],
  },
  {
    id: 'a3',
    title: 'Living with Colour Blindness',
    category: 'Lifestyle',
    readTime: '4 min read',
    coverImage: require('../../assets/access.png'),
    summary: 'Practical tips and tools to navigate daily life with color vision deficiency.',
    sections: [
      {
        type: 'paragraph',
        text: 'While there is no cure for CVD, many strategies and technologies can help you navigate daily life more effectively.',
      },
      {
        type: 'heading',
        text: 'Daily Strategies',
      },
      {
        type: 'list',
        items: [
          { title: 'Label everything', desc: 'Use text labels or symbols alongside colour coding.' },
          { title: 'Use apps', desc: 'Colour identifier apps like ReColor can name colours in real-time.' },
          { title: 'Request accommodation', desc: 'Workplaces and schools must provide accessible materials.' },
          { title: 'Smartphone accessibility', desc: 'iOS and Android both offer colour filter overlays.' },
        ],
      },
      {
        type: 'heading',
        text: 'Career Considerations',
      },
      {
        type: 'paragraph',
        text: 'Many professions have relaxed requirements for colour vision. Graphic designers, pilots, and electricians should check specific requirements, but CVD rarely prevents a fulfilling career.',
      },
    ],
  },
  {
    id: 'a4',
    title: 'Technology & Colour Accessibility',
    category: 'Technology',
    readTime: '6 min read',
    coverImage: require('../../assets/domore.png'),
    summary: 'How modern technology is breaking down barriers for people with colour vision deficiency.',
    sections: [
      {
        type: 'paragraph',
        text: 'From AI-powered assistants to GPU-accelerated simulation, technology is rapidly improving the quality of life for people with CVD.',
      },
      {
        type: 'heading',
        text: 'Key Innovations',
      },
      {
        type: 'list',
        items: [
          { title: 'Daltonization', desc: 'Algorithms that shift inaccessible colours into the visible range for each CVD type.' },
          { title: 'AI Colour Naming', desc: 'Neural networks that identify and describe colours from camera input.' },
          { title: 'CVD Simulation', desc: 'Real-time GPU shaders that let designers experience CVD conditions.' },
          { title: 'Smart Glasses', desc: 'Wearables with embedded colour enhancement lenses.' },
        ],
      },
      {
        type: 'heading',
        text: 'The Ishihara Test',
      },
      {
        type: 'paragraph',
        text: 'Developed in 1917 by Dr. Shinobu Ishihara, the pseudoisochromatic plate test remains the most widely used screening tool for red-green colour deficiency. ReColor digitises this test for broad accessibility.',
      },
    ],
  },
];
