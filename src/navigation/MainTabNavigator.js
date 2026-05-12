import { Ionicons } from "@expo/vector-icons";
import { createBottomTabNavigator } from "@react-navigation/bottom-tabs";
import * as Haptics from "expo-haptics";
import {
  Dimensions,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from "react-native";
import Animated, {
  useAnimatedStyle,
  useSharedValue,
  withSpring,
} from "react-native-reanimated";
import { useSafeAreaInsets } from "react-native-safe-area-context";

import HistoryScreen from "../screens/HistoryScreen";
import HomeScreen from "../screens/HomeScreen";
import ProfileScreen from "../screens/ProfileScreen";
import { COLORS, SHADOW } from "../theme/colors";

const Tab = createBottomTabNavigator();
const { width } = Dimensions.get("window");

const CustomTabBar = ({ state, descriptors, navigation }) => {
  const insets = useSafeAreaInsets();
  const TAB_COUNT = state.routes.length;
  const TAB_WIDTH = width / TAB_COUNT;

  // Animation for the active indicator bar at the top of the tab
  const translateX = useSharedValue(0);
  const animatedIndicatorStyle = useAnimatedStyle(() => ({
    transform: [
      {
        translateX: withSpring(state.index * TAB_WIDTH, {
          damping: 20,
          stiffness: 180,
        }),
      },
    ],
  }));

  return (
    <View style={[styles.tabBarContainer, { paddingBottom: insets.bottom }]}>
      {/* Precision Active Indicator - Top Border Style */}
      <Animated.View
        style={[
          styles.activeIndicator,
          { width: TAB_WIDTH },
          animatedIndicatorStyle,
        ]}
      />

      <View style={styles.tabContent}>
        {state.routes.map((route, index) => {
          const { options } = descriptors[route.key];
          const isFocused = state.index === index;

          const onPress = () => {
            const event = navigation.emit({
              type: "tabPress",
              target: route.key,
              canPreventDefault: true,
            });

            if (!isFocused && !event.defaultPrevented) {
              Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
              navigation.navigate(route.name);
            }
          };

          const iconMapping = {
            Home: isFocused ? "home" : "home-outline",
            History: isFocused ? "analytics" : "analytics-outline",
            Profile: isFocused ? "person" : "person-outline",
          };

          const labelMapping = {
            Home: "Home",
            History: "Records",
            Profile: "Profile",
          };

          return (
            <TouchableOpacity
              key={route.key}
              onPress={onPress}
              style={styles.tabItem}
              activeOpacity={0.7}
              accessibilityRole="button"
              accessibilityState={isFocused ? { selected: true } : {}}
              accessibilityLabel={labelMapping[route.name]}
            >
              <Ionicons
                name={iconMapping[route.name]}
                size={24}
                color={isFocused ? COLORS.primary : COLORS.textLight}
              />
              <Text
                style={[
                  styles.tabLabel,
                  {
                    color: isFocused ? COLORS.primary : COLORS.textLight,
                    fontWeight: isFocused ? "800" : "500",
                  },
                ]}
              >
                {labelMapping[route.name]}
              </Text>
            </TouchableOpacity>
          );
        })}
      </View>
    </View>
  );
};

export default function MainTabNavigator() {
  return (
    <Tab.Navigator
      tabBar={(props) => <CustomTabBar {...props} />}
      screenOptions={{
        headerShown: false,
        lazy: true, // Performance optimization
      }}
    >
      <Tab.Screen name="Home" component={HomeScreen} />
      <Tab.Screen name="History" component={HistoryScreen} />
      <Tab.Screen name="Profile" component={ProfileScreen} />
    </Tab.Navigator>
  );
}

const styles = StyleSheet.create({
  tabBarContainer: {
    position: "absolute",
    bottom: 0,
    width: "100%",
    backgroundColor: "#FFFFFF",
    borderTopWidth: 1,
    borderTopColor: "rgba(0,0,0,0.05)",
    ...SHADOW.lg, // Stronger shadow for the edge-to-edge look
  },
  tabContent: {
    flexDirection: "row",
    height: 60,
    alignItems: "center",
    justifyContent: "space-around",
  },
  tabItem: {
    flex: 1,
    height: "100%",
    alignItems: "center",
    justifyContent: "center",
    paddingTop: 8,
  },
  tabLabel: {
    fontSize: 10,
    marginTop: 4,
    letterSpacing: 0.3,
  },
  activeIndicator: {
    position: "absolute",
    top: -1, // Sits exactly on the top border
    height: 3,
    backgroundColor: COLORS.primary,
    borderBottomLeftRadius: 2,
    borderBottomRightRadius: 2,
  },
});
