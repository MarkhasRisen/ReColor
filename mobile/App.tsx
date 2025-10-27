import React from "react";
import { NavigationContainer } from "@react-navigation/native";
import { createStackNavigator } from "@react-navigation/stack";
import { LivePreviewScreen } from "./src/screens/LivePreview";
import { CalibrationScreen } from "./src/screens/Calibration";

const Stack = createStackNavigator();

export default function App() {
  return (
    <NavigationContainer>
      <Stack.Navigator initialRouteName="LivePreview">
        <Stack.Screen name="LivePreview" component={LivePreviewScreen} options={{ title: "Color Correction" }} />
        <Stack.Screen name="Calibration" component={CalibrationScreen} options={{ title: "Ishihara Test" }} />
      </Stack.Navigator>
    </NavigationContainer>
  );
}
