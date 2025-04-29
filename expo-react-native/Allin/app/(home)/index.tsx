import { Text, View, Button } from "react-native";
import Animated, 
  { useSharedValue, withSpring, withTiming, useAnimatedStyle, useAnimatedProps, Easing } from'react-native-reanimated'
import { StyleSheet } from 'react-native';
// 需要先安装 react-native-svg 依赖
// 执行: npm install react-native-svg 或 yarn add react-native-svg
import { Svg, Circle } from "react-native-svg";

const AnimatedCircle = Animated.createAnimatedComponent(Circle)

export default function Index() {
  const transform = useSharedValue<number>(0);
  const r = useSharedValue<number>(20);

  const handlePress = () => {
    transform.value = transform.value += 100;
    r.value = r.value += 10;
  };

  const animatedProps = useAnimatedProps(() => ({
    r: withTiming(r.value),
  }));

  const animatedStyle = useAnimatedStyle(() => ({
    transform: [
      { translateX: withTiming(transform.value, {
        duration: 1000,
        easing: Easing.inOut(Easing.quad),
      }) },
    ]
  }));

  return (
    <View style={styles.container}>
      <Animated.View style={[styles.animatedBox, animatedStyle]}>
        <Text>Home</Text>
      </Animated.View>
      <Svg style={styles.svg}>
        <AnimatedCircle
          cx="50%"
          cy="50%"
          r={r.value}
          fill="blue"
          animatedProps={animatedProps}
        />
      </Svg>
      <Button title="Press Me" onPress={handlePress} />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
  },
  animatedBox: {
    height: 100,
    width: 100,
    backgroundColor: "violet",
    justifyContent: "center",
    alignItems: "center",
  },
  svg: {
    height: 200,
    width: 200,
  }
});
