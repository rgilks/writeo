export interface Task {
  id: string;
  title: string;
  description: string;
  prompt: string;
  icon: string;
}

export const TASKS: Task[] = [
  {
    id: "1",
    title: "Education: Practical vs Theoretical",
    description:
      "Agree/Disagree: Some people believe universities should focus more on practical skills than theoretical knowledge. To what extent do you agree or disagree?",
    prompt:
      "Some people believe that universities should focus more on practical skills rather than theoretical knowledge. To what extent do you agree or disagree?",
    icon: "🎓",
  },
  {
    id: "2",
    title: "Technology: Social Media Impact",
    description:
      "Discussion: Some people think social media has a negative impact on society. Others believe it brings people together. Discuss both views and give your opinion.",
    prompt:
      "Some people think that social media has a negative impact on society. Others believe it brings people together and has positive effects. Discuss both views and give your own opinion.",
    icon: "📱",
  },
  {
    id: "3",
    title: "Environment: Individual vs Government",
    description:
      "Opinion: Some people think individuals should be responsible for protecting the environment. Others believe governments should take the lead. What is your view?",
    prompt:
      "Some people think that individuals should be responsible for protecting the environment. Others believe that governments should take the lead. What is your view?",
    icon: "🌍",
  },
  {
    id: "4",
    title: "Work: Remote Working",
    description:
      "Advantages/Disadvantages: More people are working from home. What are the advantages and disadvantages of this trend?",
    prompt:
      "More and more people are working from home rather than in offices. What are the advantages and disadvantages of this trend?",
    icon: "💼",
  },
  {
    id: "5",
    title: "Health: Fast Food Problem",
    description:
      "Problem/Solution: Fast food consumption is increasing worldwide. What problems does this cause, and what solutions can you suggest?",
    prompt:
      "Fast food consumption is increasing worldwide, leading to health problems. What problems does this cause, and what solutions can you suggest?",
    icon: "🍔",
  },
  {
    id: "6",
    title: "Society: Ageing Population",
    description:
      "Two-part: In many countries, the population is ageing. What are the causes of this, and what effects might it have on society?",
    prompt:
      "In many countries, the population is ageing. What are the causes of this trend, and what effects might it have on society?",
    icon: "👴",
  },
  {
    id: "7",
    title: "Culture: Global vs Local",
    description:
      "Opinion: Some people think globalization means losing local culture. Others believe it enriches culture. To what extent do you agree?",
    prompt:
      "Some people think that globalization means losing local culture and traditions. Others believe it enriches culture by bringing people together. To what extent do you agree or disagree?",
    icon: "🌐",
  },
  {
    id: "8",
    title: "Crime: Punishment vs Rehabilitation",
    description:
      "Discussion: Some people think criminals should be punished harshly. Others believe rehabilitation is more effective. Discuss both views.",
    prompt:
      "Some people think that criminals should be punished harshly to deter crime. Others believe that rehabilitation programs are more effective. Discuss both views and give your opinion.",
    icon: "⚖️",
  },
];

export const TASK_DATA: Record<string, { title: string; prompt: string }> = Object.fromEntries(
  TASKS.map((task) => [task.id, { title: task.title, prompt: task.prompt }]),
);
