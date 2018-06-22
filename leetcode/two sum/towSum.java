package com.huligang;

import java.util.HashMap;
import java.util.Map;

public class Main {

    public static int[] twoSum(int[] numbers, int target) {
        int[] result = new int[2];
        Map<Integer, Integer> map = new HashMap<>();

        for (int i = 0; i < numbers.length; i++) {
            if (map.containsKey(target - numbers[i])) {
                result[1] = i + 1;
                result[0] = map.get(target - numbers[i]);
                return result;
            }

            map.put(numbers[i], i+1);
        }

        return result;
    }

    public static void main(String[] args) {
        System.out.println("hello world");

        int [] nums = {2,7,11,15};
        int target = 9;
        int[] result = twoSum(nums, target);
        System.out.println(result[0]+", "+result[1]);
    }
}
