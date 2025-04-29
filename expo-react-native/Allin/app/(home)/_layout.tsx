import { Tabs } from "expo-router";

export default function TabsLayout() {
    return (
        <Tabs>
            <Tabs.Screen name="index" options={{ 
                headerShown: false,
                title: "Home",
            }} />
            <Tabs.Screen name="explore" options={{ headerShown: false }} />
        </Tabs>
    )
}